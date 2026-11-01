"""Frozen Q33 v2 owned-cleanup-only validator; no real processes/network/GPU."""
import argparse,importlib.util,json,pathlib,sys
sys.dont_write_bytecode=True
p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument("--candidate-dir",type=pathlib.Path,required=True);a=p.parse_args()
C=a.candidate_dir.resolve();sys.path.insert(0,str(C))
s=importlib.util.spec_from_file_location("q33_cleanup_v2",C/"speed_stage_b_v2.py");m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
results=[]
for badlog,badjob,badobserver in [(False,False,False),(True,False,False),(False,True,False),(False,False,True)]:
    trace=[];state={}
    class Stop:
        def set(self):trace.append("stop")
    class Job:
        def close(self):
            trace.append("job.close")
            if badjob:raise OSError("simulated close error")
    class Proc:
        returncode=-1
        def poll(self):trace.append("poll");return None
        def kill(self):trace.append("kill")
        def wait(self,timeout):trace.append("wait")
    class Observer:
        stop=Stop()
        def finish(self):
            trace.append("observer.finish")
            if badobserver:raise OSError("simulated observer output error")
            return {"observer_thread_exited":True}
    def event(*args,**kwargs):
        trace.append("event")
        if badlog:raise OSError("simulated logging error")
    m.cleanup_owned_v2(Job(),Proc(),Observer(),state,{"cleanup_seconds":1},event)
    checks={"stop_first":trace[0]=="stop","job_immediate":trace[1]=="job.close",
            "child_wait_before_join":trace.index("wait")<trace.index("observer.finish"),
            "no_event_before_child_wait":trace.index("event")>trace.index("wait"),
            "owned_exited":state["owned_root_exited"] is True,
            "job_close_requested":state["job_close_requested"] is True}
    assert all(checks.values()),(trace,checks)
    if badlog:assert "cleanup_log_OSError" in state["cleanup_diagnostics"]
    if badjob:assert "job_close_OSError" in state["cleanup_diagnostics"]
    if badobserver:assert state["node_timing"]["invalid_reasons"]==["observer_cleanup_OSError"]
    results.append({"bad_log":badlog,"bad_job_close":badjob,"bad_observer":badobserver,"trace":trace,"checks":checks})
print(json.dumps({"schema":"q33-cleanup-offline/v2","scenarios":results,"scenarios_completed":len(results),"all_observed":True,"no_real_processes":True,"no_network_gpu":True},indent=2))
