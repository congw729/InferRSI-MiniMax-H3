"""Q33 bounded native WS state observer. No inference imports."""
from __future__ import annotations
import asyncio
import json
import math
from pathlib import Path
import threading
import time

class BoundedJsonl:
    def __init__(self,path,max_bytes=16777216,max_messages=10000):
        self.stream=Path(path).open("x",encoding="utf-8",newline="\n")
        self.max_bytes=max_bytes;self.max_messages=max_messages;self.bytes=0;self.count=0
    def write(self,obj):
        raw=json.dumps(obj,ensure_ascii=True,allow_nan=False,separators=(",",":"))+"\n"
        size=len(raw.encode("utf-8"))
        if self.bytes+size>self.max_bytes or self.count>=self.max_messages:
            raise RuntimeError("timing journal bound")
        self.stream.write(raw);self.stream.flush();self.bytes+=size;self.count+=1
    def close(self):self.stream.close()

class NodeTimingCollector:
    def __init__(self,nodes,prompt_id,client_id):
        self.prompt_id=prompt_id;self.client_id=client_id;self.last_ns=None
        self.ready=False;self.terminal=None;self.invalid_reasons=[];self.seq=0
        self.success_received_ns=None;self.history_published_marker_received_ns=None
        self.nodes={str(k):{"class_type":v,"start_received_ns":None,"finish_received_ns":None,
                           "cached":False,"last_state":None,"first_progress_received_ns":None,
                           "last_progress_received_ns":None,"progress_value":None,"progress_max":None}
                    for k,v in nodes.items()}
    def invalidate(self,reason):
        if reason not in self.invalid_reasons:self.invalid_reasons.append(reason)
    def _node(self,node):
        if type(node) is not str or node not in self.nodes:raise RuntimeError("unknown timing node")
        return self.nodes[node]
    @staticmethod
    def _number(value):
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:
            raise RuntimeError("invalid native progress value")
        return value
    def ingest(self,message,received_ns):
        if type(received_ns) is not int or received_ns<0 or (self.last_ns is not None and received_ns<self.last_ns):
            raise RuntimeError("timing clock regression")
        self.last_ns=received_ns
        if type(message) is not dict or type(message.get("type")) is not str or type(message.get("data")) is not dict:
            raise RuntimeError("native WS message schema")
        kind=message["type"];data=message["data"];out=[]
        def emit(event,**fields):
            self.seq+=1;out.append({"sequence":self.seq,"received_perf_counter_ns":received_ns,
                "prompt_id":self.prompt_id,"event":event,**fields})
        if kind=="status":
            if "sid" in data:
                if data["sid"]!=self.client_id or self.ready:raise RuntimeError("WS sid mismatch or reconnect")
                self.ready=True;emit("observer_ready",client_id=self.client_id)
            return out
        if data.get("prompt_id")!=self.prompt_id:return out
        if not self.ready:raise RuntimeError("prompt event before WS ready")
        if kind=="execution_cached":
            ids=data.get("nodes")
            if type(ids) is not list:raise RuntimeError("cache node list")
            for node in ids:self._node(node)["cached"]=True;emit("cached",node_id=node)
        elif kind=="progress_state":
            states=data.get("nodes")
            if type(states) is not dict:raise RuntimeError("native node state map")
            for node,item in states.items():
                current=self._node(node)
                if type(item) is not dict or item.get("node_id")!=node or item.get("prompt_id")!=self.prompt_id:
                    raise RuntimeError("native state identity")
                state=item.get("state")
                if state not in {"pending","running","finished","error"}:raise RuntimeError("native lowercase state")
                self._number(item.get("value"));self._number(item.get("max"))
                if current["last_state"]==state:continue
                if current["last_state"] in {"finished","error"}:
                    raise RuntimeError("native node rerun not supported by timing interval")
                current["last_state"]=state
                if state=="running" and current["start_received_ns"] is None:current["start_received_ns"]=received_ns
                if state=="finished" and current["finish_received_ns"] is None:current["finish_received_ns"]=received_ns
                if state=="error":self.invalidate("native_node_error")
                emit("node_state",node_id=node,class_type=current["class_type"],state=state)
        elif kind=="progress":
            node=data.get("node");current=self._node(node)
            value=self._number(data.get("value"));maximum=self._number(data.get("max"))
            if maximum<value:raise RuntimeError("progress value exceeds max")
            if current["first_progress_received_ns"] is None:current["first_progress_received_ns"]=received_ns
            current["last_progress_received_ns"]=received_ns;current["progress_value"]=value;current["progress_max"]=maximum
            emit("progress",node_id=node,value=value,max=maximum)
        elif kind=="executing" and "node" in data and data["node"] is None:
            if self.terminal=="execution_success" and self.history_published_marker_received_ns is None:
                self.history_published_marker_received_ns=received_ns
                emit("history_published_marker",source="native executing(node=None) after task_done")
        elif kind in {"execution_start","execution_success","execution_error","execution_interrupted"}:
            fields={}
            timestamp=data.get("timestamp")
            if timestamp is not None:
                if type(timestamp) is not int:raise RuntimeError("native timestamp type")
                fields["native_timestamp_ms"]=timestamp
            emit(kind,**fields)
            if kind!="execution_start":
                self.terminal=kind
                if kind=="execution_success" and self.success_received_ns is None:self.success_received_ns=received_ns
                if kind!="execution_success":self.invalidate(kind)
        return out
    def should_finish(self):
        return self.terminal in {"execution_error","execution_interrupted"} or (
            self.terminal=="execution_success" and self.history_published_marker_received_ns is not None)
    def summary(self):
        nodes={}
        for node,row in self.nodes.items():
            begin,end=row["start_received_ns"],row["finish_received_ns"]
            valid=not self.invalid_reasons and not row["cached"] and begin is not None and end is not None
            nodes[node]={**row,"observed_wall_seconds":((end-begin)/1e9 if valid else None)}
        return {"schema":"q33-observed-node-timing/v1","clock":"receiver perf_counter_ns",
                "measurement":"Observed node wall interval including async WS transport; not GPU-only",
                "prompt_id":self.prompt_id,"client_id":self.client_id,"ready":self.ready,
                "terminal":self.terminal,"invalid_reasons":list(self.invalid_reasons),"nodes":nodes,
                "success_received_perf_counter_ns":self.success_received_ns,
                "history_published_marker_received_perf_counter_ns":self.history_published_marker_received_ns,
                "gpu_compute_only_seconds":None,"pure_initialization_seconds":None}

class NativeWSObserver:
    """One bounded receiver thread in the owned controller; no reconnect or dispatch."""
    def __init__(self,port,prompt_id,client_id,nodes,output_dir,portable_root,controller_started):
        self.port=port;self.collector=NodeTimingCollector(nodes,prompt_id,client_id)
        self.output_dir=Path(output_dir);self.portable_root=Path(portable_root).resolve()
        self.controller_started=controller_started
        self.stop=threading.Event();self.ready=threading.Event();self.done=threading.Event()
        self.error=None;self.thread=None;self.raw_bytes=0;self.raw_messages=0
    def start(self):
        if self.thread is not None:raise RuntimeError("observer cannot restart")
        self.output_dir.mkdir()
        self.thread=threading.Thread(target=self._thread_main,name="q33-owned-native-ws",daemon=True)
        self.thread.start()
    def _thread_main(self):
        try:asyncio.run(self._run())
        except BaseException as exc:
            self.error=type(exc).__name__;self.collector.invalidate("observer_"+type(exc).__name__)
        finally:self.done.set()
    async def _run(self):
        import aiohttp
        if not Path(aiohttp.__file__).resolve().is_relative_to(self.portable_root/"python_embeded/Lib/site-packages/aiohttp"):
            raise RuntimeError("observer aiohttp import source")
        writer=BoundedJsonl(self.output_dir/"events.jsonl")
        try:
            timeout=aiohttp.ClientTimeout(total=None,sock_connect=10)
            async with aiohttp.ClientSession(timeout=timeout,trust_env=False) as session:
                url="http://127.0.0.1:"+str(self.port)+"/ws?clientId="+self.collector.client_id
                async with session.ws_connect(url,max_msg_size=1048576,heartbeat=None) as ws:
                    connected=time.monotonic()
                    while not self.stop.is_set():
                        if not self.collector.ready and time.monotonic()-connected>10:raise RuntimeError("WS ready timeout")
                        try:message=await ws.receive(timeout=0.25)
                        except asyncio.TimeoutError:continue
                        if message.type==aiohttp.WSMsgType.TEXT:
                            self.raw_messages+=1;self.raw_bytes+=len(message.data.encode("utf-8"))
                            if self.raw_bytes>16777216 or self.raw_messages>10000:raise RuntimeError("WS receive budget")
                            rows=self.collector.ingest(json.loads(message.data),time.perf_counter_ns())
                            for row in rows:
                                row["controller_elapsed_seconds"]=time.monotonic()-self.controller_started
                                writer.write(row)
                            if self.collector.ready:self.ready.set()
                            if self.collector.should_finish():break
                        elif message.type in {aiohttp.WSMsgType.CLOSE,aiohttp.WSMsgType.CLOSED,aiohttp.WSMsgType.ERROR}:
                            self.collector.invalidate("disconnect");break
                        elif message.type==aiohttp.WSMsgType.BINARY:
                            raise RuntimeError("unexpected binary preview with previews disabled")
                    if self.collector.terminal is None:self.collector.invalidate("incomplete_or_stopped")
        finally:writer.close()
    def finish(self):
        self.stop.set()
        if self.thread is not None:self.thread.join(timeout=3)
        alive=self.thread is not None and self.thread.is_alive()
        if alive:
            # Never serialize mutable collector data while receiver still runs.
            result={"schema":"q33-observed-node-timing/v1","nodes":{},"invalid_reasons":["observer_join_timeout"],
                    "gpu_compute_only_seconds":None,"pure_initialization_seconds":None}
        else:result=self.collector.summary()
        result.update(observer_thread_exited=not alive,raw_received_bytes=self.raw_bytes,
                      raw_received_messages=self.raw_messages,error_type=self.error)
        with (self.output_dir/"summary.json").open("x",encoding="utf-8",newline="\n") as f:
            json.dump(result,f,ensure_ascii=True,allow_nan=False,indent=2);f.write("\n")
        return result
