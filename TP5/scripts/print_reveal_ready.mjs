// Owned headless renderer. Wait for Reveal's real PDF pages before printing.
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {setTimeout as delay} from 'node:timers/promises';

const [url,output,expectedText]=process.argv.slice(2);
const expected=Number(expectedText);
if(!url||!output||!expected)throw Error('Usage: node print_reveal_ready.mjs file-url output.pdf expected-pages');
const profile=await fs.mkdtemp(path.join(os.tmpdir(),'tp5-print-ready-'));
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
    try{port=Number((await fs.readFile(path.join(profile,'DevToolsActivePort'),'utf8')).split('\n')[0]);break}catch{}
    await delay(100);
  }
  if(!port)throw Error('Chrome did not start: '+errors);
  const targets=await(await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  const target=targets.find(t=>t.type==='page');
  ws=new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{ws.addEventListener('open',resolve,{once:true});ws.addEventListener('error',reject,{once:true})});
  let id=0;const pending=new Map();
  ws.addEventListener('message',event=>{const m=JSON.parse(String(event.data));if(!m.id)return;const p=pending.get(m.id);if(!p)return;pending.delete(m.id);if(m.error)p.reject(Error(JSON.stringify(m.error)));else p.resolve(m.result)});
  const call=(method,params={})=>new Promise((resolve,reject)=>{const key=++id;pending.set(key,{resolve,reject});ws.send(JSON.stringify({id:key,method,params}))});
  await call('Page.enable');await call('Runtime.enable');await call('Page.navigate',{url});
  let ready=false;
  for(let i=0;i<300;i++){
    const state=await call('Runtime.evaluate',{expression:`document.readyState==='complete' && typeof Reveal!=='undefined' && Reveal.isReady() && document.querySelectorAll('.pdf-page').length===${expected} && [...document.images].every(i=>i.complete && i.naturalWidth>0)`,returnByValue:true});
    if(state.result.value){ready=true;break;}await delay(100);
  }
  if(!ready){const debug=await call('Runtime.evaluate',{expression:`JSON.stringify({ready:document.readyState,pages:document.querySelectorAll('.pdf-page').length,images:[...document.images].filter(i=>!i.complete||!i.naturalWidth).map(i=>i.getAttribute('src'))})`,returnByValue:true});throw Error('Reveal not ready: '+debug.result.value)}
  await call('Runtime.evaluate',{expression:'document.fonts.ready.then(()=>true)',awaitPromise:true,returnByValue:true});
  const pdf=await call('Page.printToPDF',{printBackground:true,preferCSSPageSize:true,displayHeaderFooter:false});
  const bytes=Buffer.from(pdf.data,'base64');
  if(bytes.length<100000)throw Error('Unexpectedly small PDF');
  await fs.writeFile(output,bytes);
  process.stdout.write(JSON.stringify({output,bytes:bytes.length,expectedPages:expected,allImagesLoaded:true})+'\n');
}finally{
  ws?.close();
  try{process.kill(-chrome.pid,'SIGTERM')}catch{}
}
