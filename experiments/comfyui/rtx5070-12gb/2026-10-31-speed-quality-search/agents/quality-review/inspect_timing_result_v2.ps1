param([Parameter(Mandatory=$true)][string]$RunDir,[Parameter(Mandatory=$true)][string]$TrialDir,[Parameter(Mandatory=$true)][string]$ControllerDir,[Parameter(Mandatory=$true)][string]$Output)
$ErrorActionPreference='Stop'
# Descriptive post-run arithmetic audit, not a prospective acceptance validator.
function ReadJson([string]$p){if((Get-Item -LiteralPath $p).Length -gt 16777216){throw 'input bound'}; Get-Content -LiteralPath $p -Raw | ConvertFrom-Json}
function ReadRows([string]$p){if((Get-Item -LiteralPath $p).Length -gt 16777216){throw 'input bound'}; @(Get-Content -LiteralPath $p | ForEach-Object {$_|ConvertFrom-Json})}
function J($v){$v|ConvertTo-Json -Depth 35 -Compress}
$r=ReadJson (Join-Path $TrialDir 'result.json'); $s=ReadJson (Join-Path $RunDir 'timing/summary.json')
$events=ReadRows (Join-Path $RunDir 'timing/events.jsonl'); $ce=ReadRows (Join-Path $TrialDir 'events.jsonl')
$resources=ReadRows (Join-Path $TrialDir 'resources.jsonl'); $all=@($resources)+@($r.last_sample)
$graph=(ReadJson (Join-Path $ControllerDir 'graphs/original-50.json')).prompt
$canonical=(ReadJson (Join-Path $ControllerDir 'canonical/original.json')).prompt
$history=ReadJson (Join-Path $RunDir 'history.raw.json'); $h=$history.($r.prompt_id)
$checks=[ordered]@{}
$checks.result_and_summary_identical=(J $r.node_timing) -eq (J $s)
$checks.one_accepted_post=$r.prompt_posts -eq 1 -and $r.submission_outcome -eq 'accepted' -and @($ce|Where-Object event -eq 'post_attempt').Count -eq 1
$checks.observer_ready_before_post=($ce|Where-Object event -eq 'observer_ready').perf_counter_ns -lt ($ce|Where-Object event -eq 'post_attempt').perf_counter_ns
$checks.observer_identity=$s.prompt_id -eq $r.prompt_id -and $s.client_id -eq $r.client_id -and @($events|Where-Object prompt_id -ne $r.prompt_id).Count -eq 0
$checks.no_timing_invalid=$s.invalid_reasons.Count -eq 0 -and $s.error_type -eq $null -and $s.terminal -eq 'execution_success'
$checks.cleanup=$r.owned_root_exited -and $r.job_close_requested -and $r.cleanup_diagnostics.Count -eq 0 -and $s.observer_thread_exited
$checks.no_stopped_event=@($ce|Where-Object event -eq 'stopped').Count -eq 0
function EqualJson($a,$b){
 if($null -eq $a -or $null -eq $b){return $null -eq $a -and $null -eq $b}
 if($a -is [ValueType] -and $b -is [ValueType]){if($a -is [bool] -or $b -is [bool]){return $a -is [bool] -and $b -is [bool] -and $a -eq $b};return [decimal]$a -eq [decimal]$b}
 if($a -is [string] -or $b -is [string]){return $a -is [string] -and $b -is [string] -and $a -ceq $b}
 if($a -is [array] -or $b -is [array]){if(-not($a -is [array] -and $b -is [array]) -or $a.Count -ne $b.Count){return $false};for($i=0;$i -lt $a.Count;$i++){if(-not(EqualJson $a[$i] $b[$i])){return $false}};return $true}
 $ak=@($a.psobject.Properties.Name);$bk=@($b.psobject.Properties.Name);if($ak.Count -ne $bk.Count){return $false};foreach($k in $ak){if($k -cnotin $bk -or -not(EqualJson $a.$k $b.$k)){return $false}};return $true
}
$checks.actual_history_graph=EqualJson $h.prompt[2] $graph
$changed=@(foreach($p in $graph.psobject.Properties){if((J $p.Value) -ne (J $canonical.($p.Name))){$p.Name}})
$checks.only_output_prefix_changed=$changed.Count -eq 1 -and $changed[0] -eq '92'
$checks.original50=$r.profile_id -eq 'original-50' -and $graph.'9'.inputs.steps -eq 50
$checks.cached_nodes_empty=@($h.status.messages|Where-Object {$_[0] -eq 'execution_cached'})[0][1].nodes.Count -eq 0
$previous=0L; $sequence=0; $clock_ok=$true
foreach($e in $events){$sequence++;if($e.sequence -ne $sequence -or $e.received_perf_counter_ns -lt $previous){$clock_ok=$false};$previous=$e.received_perf_counter_ns}
$checks.event_sequence_clock=$clock_ok
$rows=@(foreach($p in $s.nodes.psobject.Properties){$id=$p.Name;$n=$p.Value;$beg=@($events|Where-Object {$_.node_id -eq $id -and $_.event -eq 'node_state' -and $_.state -eq 'running'});$end=@($events|Where-Object {$_.node_id -eq $id -and $_.event -eq 'node_state' -and $_.state -eq 'finished'});$delta=$null;$ok=$beg.Count -eq 1 -and $end.Count -eq 1;if($ok){$delta=($end[0].received_perf_counter_ns-$beg[0].received_perf_counter_ns)/1e9;$ok=$delta -eq $n.observed_wall_seconds -and $beg[0].received_perf_counter_ns -eq $n.start_received_ns -and $end[0].received_perf_counter_ns -eq $n.finish_received_ns -and -not $n.cached};[ordered]@{node=$id;class_type=$n.class_type;wall_seconds=$delta;event_endpoints_match=$ok}})
$checks.all_node_endpoints=@($rows|Where-Object {-not $_.event_endpoints_match}).Count -eq 0
$progress=@($events|Where-Object {$_.event -eq 'progress' -and $_.node_id -eq '14'})
$checks.fifty_ordered_progress=$progress.Count -eq 50;for($i=0;$i -lt $progress.Count;$i++){if($progress[$i].value -ne $i+1 -or $progress[$i].max -ne 50){$checks.fifty_ordered_progress=$false}}
$peakR=($all|Measure-Object owned_rss_sum_bytes -Maximum).Maximum;$peakP=($all|Measure-Object owned_private_sum_bytes -Maximum).Maximum
$minPhys=($all|Measure-Object available_physical_bytes -Minimum).Minimum;$minCommit=($all|Measure-Object available_commit_bytes -Minimum).Minimum
$gpuRows=@($all|ForEach-Object {$_.gpu_board}|Where-Object {$_});$minGpu=($gpuRows|Measure-Object free_bytes -Minimum).Minimum
$checks.resource_peaks_match=$peakR -eq $r.peak_owned_rss_sum_bytes -and $peakP -eq $r.peak_owned_private_sum_bytes
$checks.sampled_running_limits=$peakR -le $r.gates.max_owned_rss_bytes -and $peakP -le $r.gates.max_owned_private_bytes -and $minPhys -ge $r.gates.running_physical_bytes -and $minCommit -ge $r.gates.running_commit_bytes -and $minGpu -ge $r.gates.running_gpu_free_bytes
$start=($events|Where-Object event -eq 'execution_start')[0];$finish=($events|Where-Object event -eq 'execution_success')[0]
$phase=@();foreach($e in $ce){if($e.event -eq 'asset_verification_started'){$end=$ce|Where-Object {$_.event -eq 'asset_verified' -and $_.asset_id -eq $e.asset_id};$phase+=@{phase='asset_sha256';asset_id=$e.asset_id;wall_seconds=($end.perf_counter_ns-$e.perf_counter_ns)/1e9}}}
$spawn=$ce|Where-Object event -eq 'service_spawn_started';$ready=$ce|Where-Object event -eq 'dynamic_vram_confirmed';$clean=$ce|Where-Object event -eq 'owned_service_cleanup_finished'
$inputs=@();foreach($pair in @(@($TrialDir,'result.json','trial'),@($TrialDir,'events.jsonl','trial'),@($TrialDir,'resources.jsonl','trial'),@($RunDir,'timing/summary.json','run'),@($RunDir,'timing/events.jsonl','run'),@($RunDir,'history.raw.json','run'),@($ControllerDir,'graphs/original-50.json','controller'),@($ControllerDir,'canonical/original.json','controller'))){$p=Join-Path $pair[0] $pair[1];$inputs+=@{root=$pair[2];path=$pair[1];bytes=(Get-Item -LiteralPath $p).Length;sha256=(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLower()}}
$out=[ordered]@{schema='q33-independent-real-timing-audit/v2';at_utc=[DateTime]::UtcNow.ToString('o');profile='original-50';checks=$checks;failed_checks=@($checks.Keys|Where-Object {-not $checks[$_]});controller_elapsed_seconds=$r.elapsed_seconds;source_verification_seconds=($r.source_verification_end_ns-$r.source_verification_start_ns)/1e9;asset_verification=$phase;service_spawn_to_dynamic_log_confirmation_seconds=($ready.perf_counter_ns-$spawn.perf_counter_ns)/1e9;receiver_execution_interval_seconds=($finish.received_perf_counter_ns-$start.received_perf_counter_ns)/1e9;server_execution_interval_seconds=($finish.native_timestamp_ms-$start.native_timestamp_ms)/1000;node_intervals=$rows;sampler_start_to_first_step_received_seconds=($progress[0].received_perf_counter_ns-$s.nodes.'14'.start_received_ns)/1e9;progress_step1_to_step50_received_seconds=($progress[-1].received_perf_counter_ns-$progress[0].received_perf_counter_ns)/1e9;owned_cleanup_seconds=($clean.end_perf_counter_ns-$clean.start_perf_counter_ns)/1e9;resources=@{rows=$resources.Count;sampled_peak_owned_rss_bytes=$peakR;sampled_peak_owned_private_bytes=$peakP;sampled_min_available_physical_bytes=$minPhys;sampled_min_available_commit_bytes=$minCommit;sampled_min_board_free_bytes=$minGpu;board_margin_over_guard_bytes=$minGpu-$r.gates.running_gpu_free_bytes};inputs=$inputs;script_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLower();claim='INCONCLUSIVE';limits=@('Post-run arithmetic review, not a frozen prospective acceptance validator','Receiver wall includes websocket transport; sampler interval includes model load/initialization and inference','Loader node duration is not all physical GPU weight transfer','Controller main-entry clock includes source verification; old Q32 controller did not','Step 1-to-50 excludes first-step startup; it is 49 step intervals, not complete sampling duration','No GPU-only timing, full playback/audio perception or service claim','Resource extrema are sampled, not continuous; board VRAM includes other applications','No media or model weights read by this script')}
if(Test-Path -LiteralPath $Output){throw 'output exists'};[IO.File]::WriteAllText($Output,($out|ConvertTo-Json -Depth 30),[Text.UTF8Encoding]::new($false));[pscustomobject]$out|Select-Object failed_checks,controller_elapsed_seconds,server_execution_interval_seconds,resources|ConvertTo-Json -Depth 5