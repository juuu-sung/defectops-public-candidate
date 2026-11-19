import * as T from './vendor/three.module.js';
import { buildWorkbench, stations } from './workbench.js';
import { stages } from './cases.js';
import { icon } from './graphics.js';

const initial = { yaw: .46, elevation: .57, zoom: 1 };
const clamp = T.MathUtils.clamp;

export function usePlan(host, select, reason = '이 환경에서는 WebGL 3D 화면을 표시할 수 없습니다.', names = stages) {
  host.classList.add('plan-view');
  host.replaceChildren();
  const message = document.createElement('p');
  message.textContent = reason + ' 아래 작업 구역을 선택해 시연을 계속하세요.';
  const plan = document.createElement('div');
  plan.className = 'floor-plan';
  const buttons = names.map((name,i)=>{
    const b=document.createElement('button');b.type='button';b.dataset.hotspot=i;b.className='plan-zone';
    b.innerHTML=icon(['camera','target','disk','book','pen','ticket'][i],23)+'<span>'+String(i+1).padStart(2,'0')+' · '+name+'</span>';
    b.onclick=()=>select(i);plan.append(b);return b;
  });
  host.append(message,plan);
  document.querySelectorAll('[data-camera]').forEach(b=>b.disabled=true);
  document.querySelector('#scene-hint').textContent='단계를 선택해 이동';
  return {update(id,active,states){buttons.forEach((b,i)=>{
    b.classList.toggle('selected',active===i);b.dataset.state=states[i];b.setAttribute('aria-pressed',String(active===i));
  });}};
}

export function startScene(host, select, names = stages) {
  const renderer = new T.WebGLRenderer({ alpha:true,antialias:true,powerPreference:'low-power' });
  const scene = new T.Scene();
  const camera = new T.OrthographicCamera(-8,8,4,-4,.1,80);
  const view = { ...initial };
  const canvas = renderer.domElement;
  if(matchMedia('(pointer: coarse)').matches)document.querySelector('#scene-hint').textContent='가로 드래그로 회전';
  canvas.setAttribute('aria-label','검사 작업대 3D 장면. 방향키로 회전, 더하기와 빼기로 확대·축소, 0으로 시점 초기화.');
  canvas.tabIndex=0;
  host.replaceChildren(canvas);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1,2));
  renderer.setClearColor(0x000000,0);renderer.outputColorSpace=T.SRGBColorSpace;
  renderer.toneMapping=T.ACESFilmicToneMapping;renderer.toneMappingExposure=1.10;
  renderer.shadowMap.enabled=true;renderer.shadowMap.type=T.VSMShadowMap;
  renderer.shadowMap.autoUpdate=false;renderer.shadowMap.needsUpdate=true;
  scene.add(new T.HemisphereLight('#fffaf1','#aebbb0',1.65));
  const sun = new T.DirectionalLight('#fff4e3',2.5);sun.position.set(-5,12,7);sun.castShadow=true;
  Object.assign(sun.shadow.camera,{left:-8,right:8,top:7,bottom:-7,near:.2,far:34});sun.shadow.camera.updateProjectionMatrix();
  sun.shadow.mapSize.set(1024,1024);sun.shadow.radius=4;sun.shadow.blurSamples=8;
  sun.shadow.normalBias=.016;sun.shadow.bias=-.00018;scene.add(sun);
  const fill = new T.DirectionalLight('#d6e8ed',1.15);fill.position.set(9,7,-7);scene.add(fill);
  const floor=new T.Mesh(new T.PlaneGeometry(75,75),new T.ShadowMaterial({color:'#70877c',opacity:.20}));
  floor.rotation.x=-Math.PI/2;floor.position.y=-.484;floor.receiveShadow=true;scene.add(floor);
  const bench=buildWorkbench(scene);
  const labelLayer = document.createElement('div');labelLayer.className='markers';host.append(labelLayer);
  const buttons=names.map((name,i)=>{
    const b=document.createElement('button');b.type='button';b.className='marker';b.dataset.hotspot=i;
    b.innerHTML='<span class="pin-num">'+String(i+1).padStart(2,'0')+'</span><span class="pin-name">'+name+'</span>';
    b.onclick=()=>select(i);labelLayer.append(b);return b;
  });
  const target = new T.Vector3(0,.70,0), projection = new T.Vector3();
  let width=0,height=0,frame=0,lastCase=null,lost=false,backup=null,current;
  function paint(){
    frame=0;if(!width||!height||lost)return;
    const aspect=width/height, half=Math.max(3.99,6.92/aspect)/view.zoom;
    Object.assign(camera,{left:-half*aspect,right:half*aspect,top:half,bottom:-half});
    const flat=21*Math.cos(view.elevation);
    camera.position.set(Math.sin(view.yaw)*flat,21*Math.sin(view.elevation)+target.y,Math.cos(view.yaw)*flat);
    camera.lookAt(target);camera.updateProjectionMatrix();camera.updateMatrixWorld();
    renderer.render(scene,camera);
    buttons.forEach((b,i)=>{
      const [x,z,y]=stations[i];
      projection.set(x+(i===1?.05:0),y+.25,z+.06).project(camera);
      const left=(projection.x+1)*width/2,top=(1-projection.y)*height/2;
      b.style.left=left+'px';b.style.top=top+'px';
      b.hidden=left<12||left>width-12||top<10||top>height-12;
    });
  }
  function schedule(){if(!frame&&!lost)frame=requestAnimationFrame(paint);}
  function resize(){
    width=host.clientWidth;height=host.clientHeight;renderer.setSize(width,height,false);schedule();
  }
  const observer=new ResizeObserver(resize);observer.observe(host);resize();
  function cameraAction(action){
    switch(action){
      case 'left':view.yaw-=Math.PI/10;break;
      case 'right':view.yaw+=Math.PI/10;break;
      case 'up':view.elevation=clamp(view.elevation+.12,.32,1.06);break;
      case 'down':view.elevation=clamp(view.elevation-.12,.32,1.06);break;
      case 'in':view.zoom=clamp(view.zoom*1.16,.72,1.85);break;
      case 'out':view.zoom=clamp(view.zoom/1.16,.72,1.85);break;
      case 'reset':Object.assign(view,initial);break;
    }
    document.querySelector('[data-camera="in"]').disabled=view.zoom>=1.85;
    document.querySelector('[data-camera="out"]').disabled=view.zoom<=.72;schedule();
  }
  document.querySelectorAll('[data-camera]').forEach(b=>b.onclick=()=>cameraAction(b.dataset.camera));
  canvas.addEventListener('keydown',e=>{
    const action={ArrowLeft:'left',ArrowRight:'right',ArrowUp:'up',ArrowDown:'down','+':'in','=':'in','-':'out','0':'reset',Home:'reset'}[e.key];
    if(action){e.preventDefault();cameraAction(action);}
  });
  const ray=new T.Raycaster(),pointer=new T.Vector2();
  function hit(e){
    const rect=canvas.getBoundingClientRect();pointer.set(2*(e.clientX-rect.left)/width-1,1-2*(e.clientY-rect.top)/height);
    ray.setFromCamera(pointer,camera);
    let obj=ray.intersectObjects(bench.nodes,true)[0]?.object;
    while(obj&&obj.userData.station===undefined)obj=obj.parent;
    return obj?.userData.station;
  }
  let drag=null;
  canvas.addEventListener('pointerdown',e=>{
    if(!e.isPrimary||e.button!==0)return;
    drag={id:e.pointerId,x:e.clientX,y:e.clientY,yaw:view.yaw,elevation:view.elevation,moved:false};
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove',e=>{
    if(!drag||drag.id!==e.pointerId)return;
    const dx=e.clientX-drag.x,dy=e.clientY-drag.y;
    if(Math.hypot(dx,dy)>5)drag.moved=true;
    if(drag.moved){
      view.yaw=drag.yaw-dx*.009;
      if(e.pointerType!=='touch')view.elevation=clamp(drag.elevation+dy*.006,.32,1.06);
      canvas.classList.add('dragging');schedule();
    }
  });
  canvas.addEventListener('pointerup',e=>{
    if(!drag||drag.id!==e.pointerId)return;
    const clicked=!drag.moved;drag=null;canvas.classList.remove('dragging');
    if(clicked){const i=hit(e);if(i!==undefined && i<names.length)select(i);}
  });
  canvas.addEventListener('pointercancel',()=>{drag=null;canvas.classList.remove('dragging');});
  const api={update(id,active,states,decision){
    current=[id,active,states,decision];
    if(backup){backup.update(...current);return;}
    if(id!==lastCase){bench.updateMonitor(id);lastCase=id;}
    bench.setStates(active,states,decision);
    buttons.forEach((b,i)=>{
      b.classList.toggle('selected',i===active);b.dataset.state=states[i];b.setAttribute('aria-pressed',String(i===active));
      b.setAttribute('aria-label',(i+1)+'단계 '+names[i]+(states[i]==='done'?' · 완료':''));
    });
    schedule();
  }};
  canvas.addEventListener('webglcontextlost',e=>{
    e.preventDefault();lost=true;observer.disconnect();cancelAnimationFrame(frame);
    backup=usePlan(host,select,'3D 화면 연결이 중단됐습니다.',names);if(current)backup.update(...current);
  });
  return api;
}
