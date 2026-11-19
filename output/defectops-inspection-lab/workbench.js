import * as T from './vendor/three.module.js';
import { drawSpecimen } from './graphics.js';

const finish = (color, metalness = .12, roughness = .52) => new T.MeshStandardMaterial({ color, metalness, roughness });
const m = {
  ivory: finish('#f1f0e2'), warm: finish('#d9ddce'), trim: finish('#a4b0a5',.28), silver: finish('#c9d1c9',.35,.34),
  teal: finish('#207367',.20), deep: finish('#214f4b'), dark: finish('#283c3a'), rubber: finish('#354340',.04,.85),
  mint: finish('#95c6aa'), lime: finish('#c7de8d'), brass: finish('#b79964',.36,.36), amber: finish('#dd9554'),
  paper: finish('#fff7dc',.02,.9), blue: finish('#617c8a'), lens: finish('#334f5a',.45,.23),
  glow: new T.MeshStandardMaterial({color:'#c5edae',emissive:'#a3eaa0',emissiveIntensity:.42}),
  orangeLight: new T.MeshStandardMaterial({color:'#ffad56',emissive:'#f89638',emissiveIntensity:.28}),
};
const geometries = new Map();
const pooled = (key, make) => {
  if (!geometries.has(key)) geometries.set(key, make());
  return geometries.get(key);
};

// Join flat faces to rounded corners on small powder-coated and machined parts.
function softBox(w,h,d,r) {
  r = Math.min(r,w/2,h/2,d/2);
  return pooled(['box',w,h,d,r].join('/'), () => {
    if (!r) return new T.BoxGeometry(w,h,d);
    const segments=5;
    const geo=new T.BoxGeometry(2,2,2,segments,segments,segments);
    const pos=geo.attributes.position, norm=geo.attributes.normal, n=new T.Vector3();
    const arc=v=>Math.sign(v)*Math.max(0,Math.abs(v)-1/segments);
    for(let i=0;i<pos.count;i++){
      const x=pos.getX(i), y=pos.getY(i), z=pos.getZ(i);
      n.set(arc(x),arc(y),arc(z)).normalize();
      pos.setXYZ(i,Math.sign(x)*(w/2-r)+n.x*r,Math.sign(y)*(h/2-r)+n.y*r,Math.sign(z)*(d/2-r)+n.z*r);
      norm.setXYZ(i,n.x,n.y,n.z);
    }
    return geo;
  });
}
function mesh(parent,geo,mat,x,y,z) {
  const part=new T.Mesh(geo,mat);part.position.set(x,y,z);
  part.castShadow=true;part.receiveShadow=true;parent.add(part);return part;
}
function box(p,x,y,z,w,h,d,mat,r=.025){return mesh(p,softBox(w,h,d,r),mat,x,y,z);}
function drum(p,x,y,z,r,h,mat,top=r){
  return mesh(p,pooled(['cyl',r,top,h].join('/'),()=>new T.CylinderGeometry(top,r,h,32)),mat,x,y,z);
}
function rod(p,a,b,r,mat){
  const start=new T.Vector3(...a),end=new T.Vector3(...b),axis=end.clone().sub(start);
  const part=drum(p,(a[0]+b[0])/2,(a[1]+b[1])/2,(a[2]+b[2])/2,r,axis.length(),mat);
  part.quaternion.setFromUnitVectors(new T.Vector3(0,1,0),axis.normalize());return part;
}
function cable(p,pts,r,mat){
  const curve=new T.CatmullRomCurve3(pts.map(v=>new T.Vector3(...v)));
  return mesh(p,new T.TubeGeometry(curve,22,r,6,false),mat,0,0,0);
}
function group(p,x=0,y=0,z=0){const g=new T.Group();g.position.set(x,y,z);p.add(g);return g;}
function printed(p,text,x,y,z,width,height,{bg='#e9eadd',ink='#304b43',font='bold 68px monospace'}={}){
  const c=document.createElement('canvas');c.width=640;c.height=128;
  const ctx=c.getContext('2d');ctx.fillStyle=bg;ctx.fillRect(0,0,640,128);ctx.fillStyle=ink;ctx.font=font;
  ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(text,320,70,614);
  return decal(p,c,x,y,z,width,height);
}
function decal(p,canvas,x,y,z,w,h){
  const map=new T.CanvasTexture(canvas);map.colorSpace=T.SRGBColorSpace;map.anisotropy=4;
  const plane=new T.Mesh(new T.PlaneGeometry(w,h),new T.MeshBasicMaterial({map}));plane.position.set(x,y,z);p.add(plane);return plane;
}
function feet(p,w,d){
  for(const x of [-w/2,w/2])for(const z of [-d/2,d/2]) drum(p,x,.09,z,.105,.18,m.rubber);
}
function screws(p,w,d,y) {
  for(const x of [-w/2,w/2])for(const z of [-d/2,d/2]) drum(p,x,y,z,.034,.018,m.silver);
}
function bench(p,w,d,height=.69) {
  feet(p,w-.42,d-.34);
  box(p,0,.16,0,w-.19,.12,d-.15,m.deep,.05);
  box(p,0,height/2+.12,0,w-.11,height-.12,d-.09,m.teal,.10);
  box(p,0,height+.05,0,w,.12,d,m.ivory,.06);
  box(p,0,height-.07,d/2-.045,w-.24,.07,.025,m.deep);
}
function vents(p,x,y,z,n=5){
  for(let i=0;i<n;i++) box(p,x+i*.105,y,z,.045,.19,.024,m.deep,.012);
}
function tabletopChip(p,x,y,z,size=1,bent=false){
  const g=group(p,x,y,z);g.scale.setScalar(size);
  box(g,0,.055,0,.32,.10,.26,m.rubber,.026);
  box(g,0,.038,-.14,.26,.048,.12,m.silver,.012);
  for(const k of [-.1,0,.1]){
    rod(g,[k,.04,.1],[k,.04,.22],.014,m.brass);
    rod(g,[k,.04,.22],[k+(bent&&k<0?-.08:0),.03,.35],.012,m.silver);
  }
}
function cameraStation(p){
  bench(p,2.36,1.60,.64);
  box(p,-.3,.78,.10,1.43,.14,1.04,m.dark,.065);
  box(p,-.3,.85,.10,1.3,.055,.92,m.trim,.04);
  for(const x of [-.60,0])for(const z of [-.12,.28]){
    box(p,x,.884,z,.50,.025,.31,m.deep,.035);tabletopChip(p,x,.90,z-.01,.72,x<0);
  }
  box(p,.72,.82,-.43,.50,.19,.49,m.silver,.045);
  box(p,.74,1.67,-.45,.22,1.55,.23,m.warm,.026);
  box(p,.74,1.64,-.315,.067,1.42,.04,m.dark,.008);
  box(p,.18,2.36,-.4,1.34,.25,.34,m.teal,.07);
  box(p,-.31,2.30,-.13,.28,.16,.61,m.trim,.05);
  box(p,-.31,2.21,.06,.54,.52,.52,m.ivory,.075);
  for(let i=0;i<4;i++) box(p,-.50+i*.125,2.49,.02,.04,.035,.29,m.trim,.01);
  drum(p,-.31,1.89,.06,.215,.18,m.dark);drum(p,-.31,1.78,.06,.19,.11,m.brass);
  drum(p,-.31,1.70,.06,.18,.11,m.rubber);drum(p,-.31,1.634,.06,.12,.026,m.lens);
  drum(p,.9,1.93,-.41,.12,.13,m.rubber).rotation.z=Math.PI/2;
  cable(p,[[-.35,2.44,-.13],[-.1,2.68,-.47],[.73,2.58,-.56],[.91,1.3,-.56]],.030,m.dark);
  printed(p,'VISION  /  01',-.38,.41,.764,.86,.15,{bg:'#217368',ink:'#e9eddb'});
  vents(p,.43,.4,.764,5);
}
function scannerStation(p){
  bench(p,2.69,1.80,.65);
  box(p,0,.81,0,1.65,.13,1.98,m.dark,.06);
  for(let i=0;i<12;i++) rod(p,[-.65,.89,-.88+i*.16],[.65,.89,-.88+i*.16],.062,m.trim);
  [-.80,.80].forEach(x=>box(p,x,.90,.01,.09,.22,1.99,m.warm,.033));
  [-1.05,1.05].forEach(x=>{
    box(p,x,1.54,-.03,.39,1.68,1.18,m.teal,.095);
    box(p,x,1.5,.564,.27,1.38,.075,m.ivory,.04);
    box(p,x,1.07,.609,.055,.29,.028,m.trim,.008);
  });
  box(p,0,2.38,-.04,2.52,.56,1.29,m.ivory,.12);
  box(p,0,2.55,-.04,2.40,.23,1.18,m.teal,.075);
  box(p,-.04,2.36,.622,2.15,.21,.077,m.deep,.035);
  printed(p,'OPTICAL SCAN',-.22,2.36,.665,1.14,.125,{bg:'#214e48',ink:'#eef3de'});
  for(let i=0;i<3;i++) box(p,.61+i*.12,2.37,.674,.05,.06,.015,i===2?m.orangeLight:m.mint,.01);
  box(p,0,2.077,-.04,1.62,.07,.80,m.dark,.025);box(p,0,2.028,.29,1.59,.03,.10,m.glow,.012);
  for(let i=0;i<5;i++)box(p,-.62+i*.25,2.681,-.20,.12,.025,.44,m.deep,.012);
  box(p,0,.97,.12,.77,.06,.73,m.deep,.04);tabletopChip(p,0,1.015,0,1.53);
  box(p,0,1.093,.36,1.28,.013,.025,m.glow,.004);
  drum(p,1.06,2.73,-.32,.115,.17,m.dark);drum(p,1.06,2.88,-.32,.102,.19,m.orangeLight);
  drum(p,1.06,3.00,-.32,.115,.055,m.rubber);
  const glass=new T.MeshBasicMaterial({color:'#619f85',transparent:true,opacity:.12,side:T.DoubleSide,depthWrite:false});
  const light=new T.Mesh(new T.PlaneGeometry(1.52,.88),glass);light.position.set(0,1.57,.30);p.add(light);
  printed(p,'SCAN  /  02',-.60,.37,.862,.75,.13,{bg:'#217368',ink:'#e9eddb'});
  vents(p,.47,.36,.864,5);
  return { status(active, state) { glass.opacity=active ? .23 : .07;light.visible=active||state==='done'; } };
}
function reviewDesk(p){
  for(const x of [-1.10,1.10]){
    box(p,x,.46,-.06,.13,.79,1.12,m.trim,.035);
    box(p,x,.095,-.05,.30,.13,1.37,m.dark,.045);
  }
  box(p,0,.45,-.5,2.27,.10,.12,m.deep);box(p,0,.91,-.05,2.62,.16,1.49,m.ivory,.08);
  box(p,0,.81,-.04,2.50,.08,1.33,m.teal);
  box(p,.03,1.04,-.32,.89,.065,.42,m.deep,.055);box(p,.03,1.36,-.39,.17,.65,.16,m.trim);
  const display=group(p,.03,1.97,-.37);display.rotation.x=-.07;
  box(display,0,0,0,2.11,1.22,.15,m.rubber,.072);
  const canvas=document.createElement('canvas');canvas.width=620;canvas.height=320;
  const screen=decal(display,canvas,0,.018,.08,1.96,1.01);
  box(display,.8,-.557,.081,.055,.022,.009,m.lime,.005);
  box(p,-.19,1.023,.29,1.16,.073,.40,m.blue,.035);
  for(let i=0;i<10;i++)for(let j=0;j<3;j++)box(p,-.66+i*.103,1.067,.17+j*.103,.078,.022,.075,m.warm,.009);
  box(p,.72,1.048,.27,.185,.105,.31,m.dark,.048);
  cable(p,[[.7,1.04,.08],[.79,1.04,-.19],[.68,1.04,-.58],[.1,1.12,-.56]],.011,m.dark);
  // A swivel stool makes the human review station identifiable from every side.
  const chair=group(p,-.77,0,1.02);
  for(let i=0;i<4;i++){const a=i*Math.PI/2+.35;rod(chair,[0,.17,0],[.36*Math.cos(a),.115,.36*Math.sin(a)],.035,m.dark);}
  drum(chair,0,.37,0,.065,.44,m.trim);drum(chair,0,.65,0,.35,.13,m.deep);drum(chair,0,.74,0,.365,.083,m.teal);
  return { specimen(source){
    const ctx=canvas.getContext('2d');ctx.fillStyle='#203d39';ctx.fillRect(0,0,620,320);
    ctx.fillStyle='#b9d8b3';ctx.font='bold 16px monospace';ctx.fillText('DEFECTOPS   /   REVIEW',20,30);
    for(const [i,map] of [false,true].entries()){
      const x=10+i*305;
      if(typeof source==='string'){
        const sample=document.createElement('canvas');sample.width=292;sample.height=218;
        drawSpecimen(sample,source,{map});ctx.drawImage(sample,x,48);
      }else{
        ctx.fillStyle='#344f47';ctx.fillRect(x,48,292,218);
        const picture=map?source.map:source.input;
        if(picture)ctx.drawImage(picture,x+37,48,218,218);
      }
      const caption=source?.captions?.[i] ?? (map?'ANOMALY MAP':'SOURCE');
      ctx.fillStyle='#dce9d6';ctx.font='16px monospace';ctx.fillText(source?.groundTruth?caption+' + GT MASK':caption,20+i*305,296);
    }
    screen.material.map.needsUpdate=true;
  }};
}
function documentStation(p){
  box(p,0,.16,0,1.94,.19,1.33,m.deep,.08);box(p,0,.56,0,1.86,.71,1.22,m.warm,.08);
  box(p,0,.93,0,2.14,.13,1.49,m.ivory,.07);
  box(p,.64,1.20,-.15,.60,.43,.83,m.teal,.035);
  for(let i=0;i<3;i++){
    const book=group(p,.46+i*.18,1.49,-.15);book.rotation.z=i===2?-.10:0;
    box(book,0,0,0,.145,.83,.63,[m.blue,m.mint,m.deep][i],.014);
    box(book,0,.12,.321,.092,.23,.015,m.paper,.005);drum(book,0,-.18,.329,.028,.012,m.trim).rotation.x=Math.PI/2;
  }
  const folder=group(p,-.32,1.16,-.04);folder.rotation.x=-.70;
  box(folder,0,.32,0,1.02,1.28,.072,m.deep,.045);box(folder,0,.32,.052,.92,1.15,.036,m.paper,.025);
  printed(folder,'SOP   r2',0,.65,.075,.68,.19,{bg:'#fff4d7',ink:'#285b4c'});
  for(let i=0;i<4;i++)box(folder,-.035,.42-i*.108,.078,.63-(i%2)*.15,.027,.008,m.trim,.003);
  box(folder,0,.91,.08,.36,.09,.055,m.silver);
  printed(p,'SOP  /  r2',-.13,.54,.616,.86,.18);
  const bookmark=box(folder,.30,.76,.09,.12,.28,.02,m.teal,.008);
  return {status(active,state){bookmark.visible=state==='done';}};
}
function approvalStation(p){
  box(p,0,.16,0,1.76,.18,1.29,m.deep,.07);box(p,0,.41,0,1.80,.42,1.37,m.teal,.08);
  box(p,0,.66,0,2.02,.15,1.55,m.ivory,.08);
  const pad=group(p,-.12,.75,.13);pad.rotation.y=.13;
  box(pad,0,0,0,1.18,.07,.95,m.trim,.023);
  for(let i=0;i<3;i++)box(pad,i*.015-.02,.052+i*.014,0,1.07,.015,.85,m.paper,.012);
  for(let i=0;i<3;i++)box(pad,-.13,.09,-.20+i*.11,.55,.008,.019,m.trim,.002);
  const stamp=group(p,.04,.85,-.11);stamp.rotation.y=-.15;
  box(stamp,0,.06,0,.76,.13,.54,m.rubber,.065);box(stamp,0,.17,0,.83,.13,.57,m.amber,.075);
  drum(stamp,0,.31,0,.21,.18,m.brass,.14);drum(stamp,0,.50,0,.12,.25,m.deep,.15);
  drum(stamp,0,.69,0,.17,.15,m.teal,.30);drum(stamp,0,.81,0,.30,.13,m.teal,.26);drum(stamp,0,.88,0,.24,.025,m.mint);
  box(p,.63,.80,-.33,.48,.13,.43,m.dark,.05);
  printed(p,'SIGN  /  05',0,.42,.69,.83,.14,{bg:'#217368',ink:'#e9eddb'});
  const impression=printed(pad,'OK',.29,.094,.24,.29,.17,{bg:'#f6edd1',ink:'#217166'});
  impression.rotation.x=-Math.PI/2;
  const returned=printed(pad,'REV',.29,.094,.24,.29,.17,{bg:'#f6edd1',ink:'#a26635'});
  returned.rotation.x=-Math.PI/2;
  return {status(active,state,decision){impression.visible=decision==='approved';returned.visible=decision==='rejected';}};
}
function archiveStation(p){
  feet(p,1.50,.98);box(p,0,.87,0,1.82,1.56,1.21,m.teal,.10);
  box(p,0,.15,0,1.88,.13,1.24,m.deep,.045);box(p,0,1.68,0,1.94,.15,1.30,m.ivory,.065);
  for(let j=0;j<3;j++){
    const y=.44+j*.47;
    box(p,0,y,.608,1.59,.405,.075,m.deep,.035);
    box(p,0,y,.66,1.52,.35,.062,m.warm,.03);
    box(p,-.47,y+.015,.70,.30,.12,.023,m.silver,.014);box(p,-.47,y+.015,.714,.23,.064,.012,m.paper,.004);
    for(const x of [.13,.51])box(p,x,y+.02,.735,.047,.072,.077,m.deep,.013);
    box(p,.32,y+.02,.783,.42,.045,.047,m.trim,.016);
  }
  const files=group(p,-.20,1.80,.03);files.rotation.y=-.11;
  box(files,0,0,0,1.03,.09,.69,m.blue,.025);
  box(files,0,.058,0,.96,.045,.64,m.paper,.013);box(files,-.03,.09,0,.99,.045,.65,m.mint,.013);
  const ticket=group(p,.44,1.80,.23);ticket.rotation.x=-Math.PI/2;ticket.rotation.z=-.28;
  printed(ticket,'TICKET',0,0,0,.61,.28,{bg:'#fff3ca',ink:'#436050'});
  return {status(active,state){ticket.visible=state==='done';}};
}

export const stations = [
  [-3.46,-1.37,2.58], [0,-1.50,2.92], [3.43,-1.48,2.49],
  [3.48,1.57,1.74], [.18,1.62,1.59], [-3.34,1.55,1.79],
];
// Platforms carry completed/selected states independently of each machine's original material.
export function buildWorkbench(scene){
  const base=group(scene);const w=11.27,d=6.13;
  box(base,0,-.35,0,w-.14,.25,d-.12,m.trim,.13);
  box(base,0,-.22,0,w,.26,d,m.warm,.14);
  box(base,0,-.08,0,w-.09,.09,d-.09,m.ivory,.05);
  printed(base,'DEFECTOPS    /    INSPECTION LAB',-.92,-.22,d/2+.003,2.93,.145,{bg:'#d8dcce',ink:'#486158'});
  for(const x of [-4.92,4.92]) for(let i=0;i<4;i++){
    const stripe=box(base,x+i*.102,-.21,d/2+.006,.046,.13,.009,i%2?m.amber:m.deep,.002);stripe.rotation.z=-.33;
  }
  screws(base,w-.46,d-.43,-.024);
  // Flush oval pads, a common assembly aisle, and arrows describe the order around the plinth.
  const pads=[],nodes=[],machines=[];
  const builders=[cameraStation,scannerStation,reviewDesk,documentStation,approvalStation,archiveStation];
  stations.forEach(([x,z],i)=>{
    const pad=finish('#bbcabd',.15);
    box(base,x,-.010,z,3.04,.058,2.28,pad,.028);
    box(base,x,.021,z,2.93,.035,2.16,m.warm,.017);
    const node=group(base,x,.04,z);node.userData.station=i;nodes.push(node);pads.push(pad);
    machines.push(builders[i](node));
  });
  for(let i=0;i<stations.length-1;i++){
    const [ax,az]=stations[i],[bx,bz]=stations[i+1];
    const angle=Math.atan2(bx-ax,bz-az), x=(ax+bx)/2,z=(az+bz)/2;
    const arrow=group(base,x,.009,z);arrow.rotation.y=angle;
    for(const sign of [-1,1])rod(arrow,[sign*.078,0,-.065],[0,0,.035],.016,m.teal);
  }
  return {nodes,updateMonitor:id=>machines[2].specimen(id),setStates(active,states,decision){
    pads.forEach((pad,i)=>{
      pad.color.set(i===active?'#208672':states[i]==='done'?'#8cad93':'#b8c5b8');
      machines[i]?.status?.(i===active,states[i],decision);
    });
  }};
}
