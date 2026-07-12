"""[SELECTED-OUT-RING 2026-07-09] pending 路径 selector scratch 稳定槽环(v2)。

[SELECTED-PRIVATE-OUT 2026-07-07] 与 off-loop 全量私有化(refresh_rebuild_mixin
off-loop 体对七个 override 载体逐 pending 装新鲜 dict)保证了 pending result 及
其 scratch 视图(selected/bounds 等有 deferred writer 晚读)活到消费——语义
正确,但 data_ptr 每 run 新鲜:#13 STAGE-0 selector-topk captured graph 的
key(编码全部消费/产出指针)在生产 pending 路径永 miss,且
_selector_topk_graph_stable_active 看到裸 dict 恒 bypass。

本环 v2:N 个槽,每槽持有一套**持久 override 容器**(SlotStableOverrides,
形态与私有 dict 完全同=get-or-alloc,shape 键控),外加 key_norms 有效键集:

1. 私有语义不变:槽绑定 pending(selected_out_ring_slot 字段),终局唯一
   漏斗 _pending_refresh_rebuild_clear 释放前绝不重用 → 不存在"未终局
   result/scratch 被后续 run 覆写"(07-07 illegal 案毒源形态;bounds 六元组
   的 deferred 读者由同一协议覆盖)。
2. 指针稳定:同槽稳态复用同 buffer → graph key 可命中(out/bounds/ws/
   key_norms 指针全在 key 内;logf ws 不在 key 但随槽稳定,shape 变化必然
   带动 key 内字段变化 → 新 key 新捕获,烘焙指针恒一致)。
3. 内容新鲜(评审清单·快照票据):key_norms 有条件重填快路径(reallocated/
   valid_keys 门),槽容器持久会让旧内容跨 run 存活——acquire 时**清空该槽
   valid_keys 集**强制每 run 重填(pack),指针稳定与内容新鲜两全。
4. 跨流序(评审清单·原位覆写臂):begin_run 在新 run 当前流对槽存的全部
   release 事件 wait_event——produce 事件(end_run 无条件现记)管 WAW;
   pending.writer_done_event+释放点现记事件管 WAR(含 inline writer 臂)。
   换代臂:容器内 buffer 换代由 ensure 的 override 分支过 UAF 守卫。

容量不足(全部槽 busy)=spill:begin_run 返回"未接管",resolver 走旧行为
原路(off-loop 外层新鲜 dict 私有化+retention 原样生效)。[F2 对齐] spill
run 的拒捕由两个 dispatch 调用臂的门(selection_worker `_topk_ring_run`
含 `not current_run_spilled`)直接走 else-eager 实现——ptr_rebuild_miss
参数并未在这两臂传入(行为等价:spill 一样不捕不 replay)。显式计数,
非静默兜底。

线程形态:pending selector run 的解析在主流 drain 与 refresh_stream off-loop
两上下文交替、不并发(R4 守卫注释同源结论);begin_run 的 run-open 断言是
并发假设被打破时的响亮 tripwire。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

# 槽容器服务的 override 属性(与 off-loop 全量私有化清单一一对应;
# valid-keys 集单列=每 run 清空的新鲜度载体)。
SLOT_OVERRIDE_DICT_ATTRS: Tuple[str, ...] = (
    "_selector_selected_indices_out_override",
    "_selector_decode_bounds_buffers_override",
    "_selector_pipeline_workspace_override",
    "_selector_key_norms_all_cache_override",
    "_selector_key_norms_delta_buffer_override",
    "_log_f_scratch_workspace_override",
)
SLOT_VALID_SET_ATTR = "_selector_key_norms_all_valid_cache_keys_override"


class SlotStableOverrides(dict):
    """槽稳定 override 容器标记。

    与 per-run 私有 dict 的消费形态完全相同(ensure 走同一 get-or-alloc
    分支);区别仅在生命周期:随槽持久 → data_ptr 稳定。
    _selector_topk_graph_stable_active 据此放行 captured graph(裸 dict=
    per-run 新鲜指针,仍 bypass)。
    """

    __slots__ = ()


class SelectedOutRingSlot:
    __slots__ = ("index", "containers", "valid_keys", "release_events", "busy")

    def __init__(self, index: int) -> None:
        self.index = int(index)
        self.containers: Dict[str, SlotStableOverrides] = {
            attr: SlotStableOverrides() for attr in SLOT_OVERRIDE_DICT_ATTRS
        }
        self.valid_keys: set = set()
        self.release_events: Tuple[Any, ...] = tuple()
        self.busy: bool = False

    def first_cuda_device(self) -> Optional[torch.device]:
        for container in self.containers.values():
            for value in container.values():
                if isinstance(value, torch.Tensor):
                    if value.is_cuda:
                        return value.device
                elif isinstance(value, tuple):
                    for item in value:
                        if isinstance(item, torch.Tensor) and item.is_cuda:
                            return item.device
        return None


class SelectedOutRing:
    __slots__ = (
        "_slots",
        "_run_open",
        "_current_slot",
        "spill_run_count",
        "acquire_count",
        "release_count",
    )

    def __init__(self, slots: int) -> None:
        self._slots = tuple(SelectedOutRingSlot(i) for i in range(max(1, int(slots))))
        self._run_open = False
        self._current_slot: Optional[SelectedOutRingSlot] = None
        self.spill_run_count = 0
        self.acquire_count = 0
        self.release_count = 0

    # ------------------------------------------------------------------
    @property
    def current_run_spilled(self) -> bool:
        """True iff 本 run 未被槽接管(容量满 spill)。

        dispatch 以此走 ptr_rebuild_miss:spill run 的指针是外层私有 dict
        的瞬时指针,不捕图(captured result 会把瞬时 buffer 持引成永久
        泄漏,且 key 永不复现=纯图污染)。
        """
        return bool(self._run_open and self._current_slot is None)

    @property
    def run_open(self) -> bool:
        return bool(self._run_open)

    def begin_run(self, *, preferred: int = -1) -> Optional[SelectedOutRingSlot]:
        """开 run 窗口并占用一个空闲槽;容量满返回 None(spill=旧行为原路)。

        preferred(通常传 pending.buf_id):优先取 preferred%N 号槽——把槽
        与 capture ring buf 稳定配对,graph key 组合数从 槽数×环深 收敛到
        ≈环深(否则组合数撞 8-graph 上限触发 clear-all,replay 永 0;
        rv2g 取证实锤=capture 64/replay 0)。占用时:在当前流(=本 run 的
        发射流)对槽存的全部 release 事件 wait_event(WAW/WAR 序),并清空
        该槽 key_norms valid 集(强制本 run 重填=内容新鲜)。
        """
        if self._run_open:
            raise RuntimeError(
                "selected-out ring: begin_run while a run is open — pending "
                "selector runs are alternate-not-concurrent by design; a nested "
                "or concurrent resolve breaks the ring protocol (fail-fast)"
            )
        self._run_open = True
        chosen: Optional[SelectedOutRingSlot] = None
        if preferred >= 0:
            _pref_slot = self._slots[int(preferred) % len(self._slots)]
            if not _pref_slot.busy:
                chosen = _pref_slot
        if chosen is None:
            for slot in self._slots:
                if not slot.busy:
                    chosen = slot
                    break
        if chosen is None:
            self.spill_run_count += 1
            self._current_slot = None
            return None
        if chosen.release_events:
            device = chosen.first_cuda_device()
            if device is not None:
                cur = torch.cuda.current_stream(device=device)
                for evt in chosen.release_events:
                    cur.wait_event(evt)
        chosen.release_events = tuple()
        chosen.valid_keys.clear()
        chosen.busy = True
        self._current_slot = chosen
        self.acquire_count += 1
        return chosen

    def end_run(self) -> Optional[SelectedOutRingSlot]:
        """关 run 窗口;返回占用槽(spill=None)。

        对占用槽在当前流无条件现记 produce 事件:下任 begin_run 的 WAW 序
        不依赖条件创建的 selector_done_event。
        """
        if not self._run_open:
            raise RuntimeError("selected-out ring: end_run without an open run")
        slot = self._current_slot
        if slot is not None:
            device = slot.first_cuda_device()
            if device is not None:
                evt = torch.cuda.Event()
                evt.record(torch.cuda.current_stream(device=device))
                slot.release_events = slot.release_events + (evt,)
        self._run_open = False
        self._current_slot = None
        return slot

    def release(
        self, slot: Optional[SelectedOutRingSlot], *, writer_done_event: Any = None
    ) -> None:
        """终局释放(幂等)。存入下任 begin_run 须等的消费序事件。

        writer_done_event=pending 的 writer 完成事件(可 None:drop 或
        inline writer 臂);释放点当前流现记事件兜住 inline 消费序——释放
        点所在流已按既有纪律 wait 过该 pending 的 async 事件(drop/accept
        扫描),现记事件把这一序固化给下任。
        """
        if slot is None:
            return
        if not slot.busy:
            return
        evts = list(slot.release_events)
        if writer_done_event is not None:
            evts.append(writer_done_event)
        device = slot.first_cuda_device()
        if device is not None:
            evt = torch.cuda.Event()
            evt.record(torch.cuda.current_stream(device=device))
            evts.append(evt)
        slot.release_events = tuple(evts)
        slot.busy = False
        self.release_count += 1
