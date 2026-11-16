import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {setTimeout as delay} from 'node:timers/promises';

const [url,output]=process.argv.slice(2);
if(!url||!output)throw Error('Usage: node check_nsynth_listening.mjs file-url output-dir');
await fs.mkdir(output,{recursive:true});
const profile=await fs.mkdtemp(path.join(os.tmpdir(),'tp5-listen-'));
const chrome=spawn('/snap/bin/chromium',['--headless','--no-sandbox','--disable-gpu','--no-first-run',
  '--remote-debugging-port=0','--remote-allow-origins=*','--allow-file-access-from-files',
  '--user-data-dir='+profile,'about:blank'],{detached:true,stdio:['ignore','ignore','pipe']});
let errors='';chrome.stderr.on('data',b=>{errors=(errors+b.toString()).slice(-6000)});
let ws;
try{
  let port;
  for(let i=0;i<200;i++){
    const endpoint=errors.match(/DevTools listening on ws:\/\/127\.0\.0\.1:(\d+)/);
    if(endpoint){port=Number(endpoint[1]);break;}
    await delay(100);
  }
  if(!port)throw Error('Chrome did not start: '+errors);
  const targets=await(await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  ws=new WebSocket(targets.find(t=>t.type==='page').webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true})});
  let id=0;const pending=new Map();
  ws.addEventListener('message',event=>{const m=JSON.parse(String(event.data));if(!m.id)return;const p=pending.get(m.id);if(!p)return;pending.delete(m.id);if(m.error)p.reject(Error(JSON.stringify(m.error)));else p.resolve(m.result)});
  const call=(method,params={})=>new Promise((resolve,reject)=>{const key=++id;pending.set(key,{resolve,reject});ws.send(JSON.stringify({id:key,method,params}))});
  const evaluate=async expression=>{const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true,userGesture:true});if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value};
  await call('Page.enable');await call('Runtime.enable');
  await call('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false});
  await call('Page.navigate',{url});
  let ready=false;
  for(let i=0;i<150;i++){if(await evaluate("document.readyState==='complete'&&document.querySelectorAll('audio').length===24")){ready=true;break}await delay(100)}
  if(!ready)throw Error('Listening page did not load');
  const structure=await evaluate("({notes:document.querySelectorAll('.note').length,audio:document.querySelectorAll('audio').length,embedded:[...document.querySelectorAll('audio')].every(a=>a.src.startsWith('data:audio/wav;base64,'))})");
  const coverage=await evaluate(`(()=>{const results=[];const seed=document.getElementById('seed'),level=document.getElementById('sigma');for(const s of ['7','17','27'])for(const l of ['0.0','0.05','0.1','0.2','0.3','0.4','0.5']){seed.value=s;level.value=l;level.dispatchEvent(new Event('change'));const a=[...document.querySelectorAll('audio')];results.push({seed:s,sigma:l,count:a.length,embedded:a.every(x=>x.src.startsWith('data:audio/wav;base64,')),zero_has_identical_clean_and_input:l==='0.0'?Array.from({length:6},(_,i)=>a[4*i].src===a[4*i+1].src).every(Boolean):true})}seed.value='7';level.value='0.1';level.dispatchEvent(new Event('change'));return results})()`);
  if(coverage.length!==21||!coverage.every(r=>r.count===24&&r.embedded&&r.zero_has_identical_clean_and_input))throw Error('Missing or misaligned seed/noise combination '+JSON.stringify(coverage));
  const selection=await evaluate(`(()=>{const initial=[...document.querySelectorAll('audio')].slice(0,4).map(a=>a.src);const seed=document.getElementById('seed');seed.value='17';seed.dispatchEvent(new Event('change'));const changed=[...document.querySelectorAll('audio')].slice(0,4).map(a=>a.src);const phase=document.getElementById('sigma');phase.value='0.2';phase.dispatchEvent(new Event('change'));const random=[...document.querySelectorAll('audio')].slice(0,4).map(a=>a.src);const result={seed_changes_models:initial[2]!==changed[2]&&initial[3]!==changed[3],seed_keeps_clean_reference:initial[0]===changed[0],seed_changes_noise:initial[1]!==changed[1],level_changes_both_models_and_input:[1,2,3].every(i=>changed[i]!==random[i]),level_keeps_clean_reference:changed[0]===random[0]};seed.value='7';phase.value='0.1';phase.dispatchEvent(new Event('change'));return result})()`);
  const playback=await evaluate(`(async()=>{let a=[...document.querySelectorAll('audio')];const started=async audio=>{await audio.play();for(let i=0;i<60&&audio.currentTime===0;i++)await new Promise(r=>setTimeout(r,100))};a[2].loop=true;await started(a[2]);const first={time:a[2].currentTime,paused:a[2].paused,duration:a[2].duration,error:a[2].error};await started(a[3]);const next={time:a[3].currentTime,previousPaused:a[2].paused,previousTime:a[2].currentTime,error:a[3].error};document.getElementById('stop').click();return {first,next,allStopped:a.every(x=>x.paused&&x.currentTime===0)}})()`);
  if(!Object.values(selection).every(Boolean)||playback.first.paused||playback.first.time<=0||playback.next.time<=0||!playback.next.previousPaused||!playback.allStopped)throw Error('Audio/control test failed '+JSON.stringify({selection,playback}));
  await fs.writeFile(path.join(output,'nsynth_denoising_pghi_desktop.png'),Buffer.from((await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false})).data,'base64'));
  await call('Emulation.setDeviceMetricsOverride',{width:390,height:1050,deviceScaleFactor:1,mobile:true});
  const mobile=await evaluate('({width:innerWidth,bodyWidth:document.documentElement.scrollWidth,players:document.querySelectorAll("audio").length})');
  if(mobile.bodyWidth>mobile.width)throw Error('Horizontal overflow on mobile');
  await fs.writeFile(path.join(output,'nsynth_denoising_pghi_mobile.png'),Buffer.from((await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false})).data,'base64'));
  const result={structure,coverage,selection,playback,mobile};
  await fs.writeFile(path.join(output,'browser_audit.json'),JSON.stringify(result,null,2));
  process.stdout.write(JSON.stringify(result)+'\n');
}finally{ws?.close();try{process.kill(-chrome.pid,'SIGTERM')}catch{}}


