var ne=Object.defineProperty;var ie=(e,t,r)=>t in e?ne(e,t,{enumerable:!0,configurable:!0,writable:!0,value:r}):e[t]=r;var m=(e,t,r)=>(ie(e,typeof t!="symbol"?t+"":t,r),r);(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const n of document.querySelectorAll('link[rel="modulepreload"]'))i(n);new MutationObserver(n=>{for(const s of n)if(s.type==="childList")for(const l of s.addedNodes)l.tagName==="LINK"&&l.rel==="modulepreload"&&i(l)}).observe(document,{childList:!0,subtree:!0});function r(n){const s={};return n.integrity&&(s.integrity=n.integrity),n.referrerPolicy&&(s.referrerPolicy=n.referrerPolicy),n.crossOrigin==="use-credentials"?s.credentials="include":n.crossOrigin==="anonymous"?s.credentials="omit":s.credentials="same-origin",s}function i(n){if(n.ep)return;n.ep=!0;const s=r(n);fetch(n.href,s)}})();var re=`struct VSOutput {
  @builtin(position) position: vec4f,
  @location(0) posWorld: vec3f,
  @location(1) normalWorld: vec3f,
  @location(2) tangentWorld: vec3f,
  @location(3) texCoord: vec2f,
};

struct Vertex {
  @location(0) pos: vec3f,
  @location(1) norm: vec3f,
  @location(2) tangent: vec3f,
  @location(3) tex: vec2f,
};

struct Uniforms {
  model: mat4x4f,
  view: mat4x4f,
  projection: mat4x4f,
  invTransposedModel: mat4x4f
};

@group(0) @binding(0) var<uniform> uni: Uniforms;

@vertex fn vs(
  input: Vertex,
) -> VSOutput {
  var output: VSOutput;
  
  output.normalWorld = normalize(uni.invTransposedModel * vec4f(input.norm, 1.0)).xyz;
  output.tangentWorld = normalize(uni.model * vec4f(input.tangent, 1.0)).xyz;
  output.texCoord = input.tex;

	output.posWorld = (uni.model * vec4f(input.pos, 1.0)).xyz;
  output.position = uni.projection * uni.view * uni.model * vec4f(input.pos , 1.0);
  return output;
}`,se=`struct VSOutput {
  @builtin(position) position: vec4f,
  @location(0) posWorld: vec3f,
  @location(1) normalWorld: vec3f,
  @location(2) tangentWorld: vec3f,
  @location(3) texCoord: vec2f,
};
const PI: f32 = 3.141592653589793;

fn DistributionGGX(N: vec3f, H: vec3f, roughness: f32) -> f32 {
    let a: f32 = roughness * roughness;
    let a2: f32 = a * a;
    let NdotH: f32 = max(dot(N, H), 0.0);
    let NdotH2: f32 = NdotH * NdotH;

    let nom: f32 = a2;
    let denom: f32 = (NdotH2 * (a2 - 1.0) + 1.0);
    let denomFinal: f32 = PI * denom * denom;

    return nom / denomFinal;
}

fn GeometrySchlickGGX(NdotV: f32, roughness: f32) -> f32 {
    let r: f32 = (roughness + 1.0);
    let k: f32 = (r * r) / 8.0;

    let nom: f32 = NdotV;
    let denom: f32 = NdotV * (1.0 - k) + k;

    return nom / denom;
}

fn GeometrySmith(N: vec3f, V: vec3f, L: vec3f, roughness: f32) -> f32 {
    let NdotV: f32 = max(dot(N, V), 0.0);
    let NdotL: f32 = max(dot(N, L), 0.0);
    let ggx2: f32 = GeometrySchlickGGX(NdotV, roughness);
    let ggx1: f32 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

fn fresnelSchlick(cosTheta: f32, F0: vec3f) -> vec3f {
    return F0 + (vec3f(1.0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

fn fresnelSchlickRoughness(cosTheta: f32, F0: vec3f, roughness: f32) -> vec3f {
    return F0 + (max(vec3f(1.0 - roughness, 1.0 - roughness, 1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

struct Uniforms {
  camPos: vec3f,
  padding: f32
};

@group(0) @binding(1) var<uniform> uni: Uniforms;
@group(0) @binding(2) var sampler2D: sampler;

@group(0) @binding(3) var albedoMap: texture_2d<f32>;
@group(0) @binding(4) var normalMap: texture_2d<f32>;
@group(0) @binding(5) var metallicMap: texture_2d<f32>;
@group(0) @binding(6) var roughnessMap: texture_2d<f32>;
@group(0) @binding(7) var aoMap: texture_2d<f32>;
@group(0) @binding(8) var brdfMap: texture_2d<f32>;

@fragment fn fs(input: VSOutput) -> @location(0) vec4f {
  
  let albedo:vec3f = pow(textureSample(albedoMap, sampler2D, input.texCoord).rgb, vec3(2.2));
  let metallic:f32 = textureSample(metallicMap, sampler2D, input.texCoord).r;
  let roughness:f32 = textureSample(roughnessMap, sampler2D, input.texCoord).r;
  let ao:f32 = textureSample(aoMap, sampler2D, input.texCoord).r;

  let normalWorld:vec3f = normalize(input.normalWorld);
  
  
  let tangent:vec3f = normalize(input.tangentWorld - dot(input.tangentWorld, normalWorld) * normalWorld);
  let bitangent:vec3f = cross(normalWorld, tangent);
  let TBN:mat3x3f = mat3x3f(tangent, bitangent, normalWorld);
  let N:vec3f = normalize(TBN * (textureSample(normalMap, sampler2D, input.texCoord).xyz * 2.0 - 1.0));

  let V:vec3f = normalize(uni.camPos - input.posWorld);
  let R:vec3f = reflect(-V, N);

  let lightRadiances:vec3f = vec3f(10.0);

  var F0:vec3f = vec3f(0.04);
  F0 = mix(F0, albedo, metallic);

  
  let L:vec3f = normalize(uni.camPos - input.posWorld);
  let H:vec3f = normalize(V + L);
  let distance:f32 = length(uni.camPos - input.posWorld);
  let attenuation:f32 = 1.0 / (distance * distance);
  let radiance:vec3f = lightRadiances * attenuation;

  let F:vec3f = fresnelSchlick(max(dot(H, V), 0.0), F0);
  let G:f32 = GeometrySmith(N, V, L, roughness);
  let NDF:f32 = DistributionGGX(N, H, roughness);

  let numerator:vec3f = NDF * G * F;
  
  let denominator:f32 = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
  let specular:vec3f = numerator / denominator;

  let kS:vec3f = F;

  let kD:vec3f = (vec3(1.0) - kS) * (1.0 - metallic);

  
  let NdotL:f32 = max(dot(N, L), 0.0);

  let directLight:vec3f = (kD * albedo / PI + specular) * radiance * NdotL;

  
  

  
  
  

  
  
  

  
  
  
  
  
  

  

  var color:vec3f = directLight;

  
  color = color / (color + vec3(1.0));
  
  color = pow(color, vec3(1.0/2.2));

  return vec4f(color, 1.0);

  
  
}`;class le{constructor(t){m(this,"_canvas");m(this,"_adapter");m(this,"_device");m(this,"_context");m(this,"_encoder");m(this,"_pass");m(this,"_depthTexture");this._canvas=t}get device(){return this._device}async initialize(t,r){var i,n,s;if(this._canvas.width=t,this._canvas.height=r,this._adapter=await((i=navigator.gpu)==null?void 0:i.requestAdapter()),this._device=await((n=this._adapter)==null?void 0:n.requestDevice()),!this._device){const l="Your device does not support WebGPU.";console.error(l);const c=document.createElement("p");c.innerHTML=l,(s=document.querySelector("body"))==null||s.prepend(c)}this._context=this._canvas.getContext("webgpu"),this._context.configure({device:this._device,format:navigator.gpu.getPreferredCanvasFormat()})}getPass(){if(!this._context||!this._device){console.error("RenderEnv was not initialized!");return}const t=this._context.getCurrentTexture();(!this._depthTexture||this._depthTexture.width!==t.width||this._depthTexture.height!==t.height)&&(this._depthTexture&&this._depthTexture.destroy(),this._depthTexture=this._device.createTexture({size:[t.width,t.height],format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT}));const r={colorAttachments:[{view:t.createView(),clearValue:[0,0,0,1],loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:this._depthTexture.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}};return this._encoder=this._device.createCommandEncoder(),this._pass=this._encoder.beginRenderPass(r),this._pass}endPass(){var r,i;if(!this._device){console.error("RenderEnv was not initialized!");return}(r=this._pass)==null||r.end();const t=(i=this._encoder)==null?void 0:i.finish();t&&this._device.queue.submit([t])}}class ce{constructor({label:t,device:r,vertexShader:i,fragmentShader:n,buffer:s,bindGroupLayouts:l}){m(this,"_label");m(this,"_pipeline");this._label=t,this._pipeline=r.createRenderPipeline({label:`${t} pipeline`,layout:l?r.createPipelineLayout({bindGroupLayouts:l}):"auto",vertex:{module:r.createShaderModule({label:`${t} vertex shader`,code:i}),buffers:s??[{arrayStride:11*Float32Array.BYTES_PER_ELEMENT,attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:3*Float32Array.BYTES_PER_ELEMENT,format:"float32x3"},{shaderLocation:2,offset:6*Float32Array.BYTES_PER_ELEMENT,format:"float32x3"},{shaderLocation:3,offset:9*Float32Array.BYTES_PER_ELEMENT,format:"float32x2"}]}]},fragment:{module:r.createShaderModule({label:`${t} fragment shader`,code:n}),targets:[{format:navigator.gpu.getPreferredCanvasFormat()}]},primitive:{topology:"triangle-list",cullMode:"back"},depthStencil:{depthWriteEnabled:!0,depthCompare:"less",format:"depth24plus"}})}getBindGroupLayout(t){return this._pipeline.getBindGroupLayout(t)}use(t){if(!t){console.error(`GPURenderPassEncoder was not passed to ${this._label} pipeline`);return}t.setPipeline(this._pipeline)}}const C=1e-6;let T=typeof Float32Array<"u"?Float32Array:Array;const ae=Math.PI/180;function Y(e){return e*ae}function oe(){let e=new T(9);return T!=Float32Array&&(e[1]=0,e[2]=0,e[3]=0,e[5]=0,e[6]=0,e[7]=0),e[0]=1,e[4]=1,e[8]=1,e}function R(){let e=new T(16);return T!=Float32Array&&(e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[11]=0,e[12]=0,e[13]=0,e[14]=0),e[0]=1,e[5]=1,e[10]=1,e[15]=1,e}function fe(e){let t=new T(16);return t[0]=e[0],t[1]=e[1],t[2]=e[2],t[3]=e[3],t[4]=e[4],t[5]=e[5],t[6]=e[6],t[7]=e[7],t[8]=e[8],t[9]=e[9],t[10]=e[10],t[11]=e[11],t[12]=e[12],t[13]=e[13],t[14]=e[14],t[15]=e[15],t}function de(e){return e[0]=1,e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=1,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[10]=1,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function he(e,t){if(e===t){let r=t[1],i=t[2],n=t[3],s=t[6],l=t[7],c=t[11];e[1]=t[4],e[2]=t[8],e[3]=t[12],e[4]=r,e[6]=t[9],e[7]=t[13],e[8]=i,e[9]=s,e[11]=t[14],e[12]=n,e[13]=l,e[14]=c}else e[0]=t[0],e[1]=t[4],e[2]=t[8],e[3]=t[12],e[4]=t[1],e[5]=t[5],e[6]=t[9],e[7]=t[13],e[8]=t[2],e[9]=t[6],e[10]=t[10],e[11]=t[14],e[12]=t[3],e[13]=t[7],e[14]=t[11],e[15]=t[15];return e}function pe(e,t){let r=t[0],i=t[1],n=t[2],s=t[3],l=t[4],c=t[5],a=t[6],d=t[7],o=t[8],h=t[9],f=t[10],p=t[11],u=t[12],v=t[13],g=t[14],x=t[15],b=r*c-i*l,M=r*a-n*l,E=r*d-s*l,P=i*a-n*c,w=i*d-s*c,G=n*d-s*a,y=o*v-h*u,S=o*g-f*u,N=o*x-p*u,z=h*g-f*v,F=h*x-p*v,B=f*x-p*g,_=b*B-M*F+E*z+P*N-w*S+G*y;return _?(_=1/_,e[0]=(c*B-a*F+d*z)*_,e[1]=(n*F-i*B-s*z)*_,e[2]=(v*G-g*w+x*P)*_,e[3]=(f*w-h*G-p*P)*_,e[4]=(a*N-l*B-d*S)*_,e[5]=(r*B-n*N+s*S)*_,e[6]=(g*E-u*G-x*M)*_,e[7]=(o*G-f*E+p*M)*_,e[8]=(l*F-c*N+d*y)*_,e[9]=(i*N-r*F-s*y)*_,e[10]=(u*w-v*E+x*b)*_,e[11]=(h*E-o*w-p*b)*_,e[12]=(c*S-l*z-a*y)*_,e[13]=(r*z-i*S+n*y)*_,e[14]=(v*M-u*P-g*b)*_,e[15]=(o*P-h*M+f*b)*_,e):null}function ge(e,t,r){let i=r[0],n=r[1],s=r[2],l,c,a,d,o,h,f,p,u,v,g,x;return t===e?(e[12]=t[0]*i+t[4]*n+t[8]*s+t[12],e[13]=t[1]*i+t[5]*n+t[9]*s+t[13],e[14]=t[2]*i+t[6]*n+t[10]*s+t[14],e[15]=t[3]*i+t[7]*n+t[11]*s+t[15]):(l=t[0],c=t[1],a=t[2],d=t[3],o=t[4],h=t[5],f=t[6],p=t[7],u=t[8],v=t[9],g=t[10],x=t[11],e[0]=l,e[1]=c,e[2]=a,e[3]=d,e[4]=o,e[5]=h,e[6]=f,e[7]=p,e[8]=u,e[9]=v,e[10]=g,e[11]=x,e[12]=l*i+o*n+u*s+t[12],e[13]=c*i+h*n+v*s+t[13],e[14]=a*i+f*n+g*s+t[14],e[15]=d*i+p*n+x*s+t[15]),e}function ue(e,t,r){let i=r[0],n=r[1],s=r[2];return e[0]=t[0]*i,e[1]=t[1]*i,e[2]=t[2]*i,e[3]=t[3]*i,e[4]=t[4]*n,e[5]=t[5]*n,e[6]=t[6]*n,e[7]=t[7]*n,e[8]=t[8]*s,e[9]=t[9]*s,e[10]=t[10]*s,e[11]=t[11]*s,e[12]=t[12],e[13]=t[13],e[14]=t[14],e[15]=t[15],e}function X(e,t,r,i){let n=i[0],s=i[1],l=i[2],c=Math.sqrt(n*n+s*s+l*l),a,d,o,h,f,p,u,v,g,x,b,M,E,P,w,G,y,S,N,z,F,B,_,D;return c<C?null:(c=1/c,n*=c,s*=c,l*=c,a=Math.sin(r),d=Math.cos(r),o=1-d,h=t[0],f=t[1],p=t[2],u=t[3],v=t[4],g=t[5],x=t[6],b=t[7],M=t[8],E=t[9],P=t[10],w=t[11],G=n*n*o+d,y=s*n*o+l*a,S=l*n*o-s*a,N=n*s*o-l*a,z=s*s*o+d,F=l*s*o+n*a,B=n*l*o+s*a,_=s*l*o-n*a,D=l*l*o+d,e[0]=h*G+v*y+M*S,e[1]=f*G+g*y+E*S,e[2]=p*G+x*y+P*S,e[3]=u*G+b*y+w*S,e[4]=h*N+v*z+M*F,e[5]=f*N+g*z+E*F,e[6]=p*N+x*z+P*F,e[7]=u*N+b*z+w*F,e[8]=h*B+v*_+M*D,e[9]=f*B+g*_+E*D,e[10]=p*B+x*_+P*D,e[11]=u*B+b*_+w*D,t!==e&&(e[12]=t[12],e[13]=t[13],e[14]=t[14],e[15]=t[15]),e)}function J(e,t,r){let i=Math.sin(r),n=Math.cos(r),s=t[4],l=t[5],c=t[6],a=t[7],d=t[8],o=t[9],h=t[10],f=t[11];return t!==e&&(e[0]=t[0],e[1]=t[1],e[2]=t[2],e[3]=t[3],e[12]=t[12],e[13]=t[13],e[14]=t[14],e[15]=t[15]),e[4]=s*n+d*i,e[5]=l*n+o*i,e[6]=c*n+h*i,e[7]=a*n+f*i,e[8]=d*n-s*i,e[9]=o*n-l*i,e[10]=h*n-c*i,e[11]=f*n-a*i,e}function Q(e,t,r){let i=Math.sin(r),n=Math.cos(r),s=t[0],l=t[1],c=t[2],a=t[3],d=t[8],o=t[9],h=t[10],f=t[11];return t!==e&&(e[4]=t[4],e[5]=t[5],e[6]=t[6],e[7]=t[7],e[12]=t[12],e[13]=t[13],e[14]=t[14],e[15]=t[15]),e[0]=s*n-d*i,e[1]=l*n-o*i,e[2]=c*n-h*i,e[3]=a*n-f*i,e[8]=s*i+d*n,e[9]=l*i+o*n,e[10]=c*i+h*n,e[11]=a*i+f*n,e}function ve(e,t,r){let i=Math.sin(r),n=Math.cos(r),s=t[0],l=t[1],c=t[2],a=t[3],d=t[4],o=t[5],h=t[6],f=t[7];return t!==e&&(e[8]=t[8],e[9]=t[9],e[10]=t[10],e[11]=t[11],e[12]=t[12],e[13]=t[13],e[14]=t[14],e[15]=t[15]),e[0]=s*n+d*i,e[1]=l*n+o*i,e[2]=c*n+h*i,e[3]=a*n+f*i,e[4]=d*n-s*i,e[5]=o*n-l*i,e[6]=h*n-c*i,e[7]=f*n-a*i,e}function me(e,t,r,i,n){const s=1/Math.tan(t/2);if(e[0]=s/r,e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=s,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[11]=-1,e[12]=0,e[13]=0,e[15]=0,n!=null&&n!==1/0){const l=1/(i-n);e[10]=(n+i)*l,e[14]=2*n*i*l}else e[10]=-1,e[14]=-2*i;return e}const _e=me;function xe(e,t,r,i){let n,s,l,c,a,d,o,h,f,p,u=t[0],v=t[1],g=t[2],x=i[0],b=i[1],M=i[2],E=r[0],P=r[1],w=r[2];return Math.abs(u-E)<C&&Math.abs(v-P)<C&&Math.abs(g-w)<C?de(e):(o=u-E,h=v-P,f=g-w,p=1/Math.sqrt(o*o+h*h+f*f),o*=p,h*=p,f*=p,n=b*f-M*h,s=M*o-x*f,l=x*h-b*o,p=Math.sqrt(n*n+s*s+l*l),p?(p=1/p,n*=p,s*=p,l*=p):(n=0,s=0,l=0),c=h*l-f*s,a=f*n-o*l,d=o*s-h*n,p=Math.sqrt(c*c+a*a+d*d),p?(p=1/p,c*=p,a*=p,d*=p):(c=0,a=0,d=0),e[0]=n,e[1]=c,e[2]=o,e[3]=0,e[4]=s,e[5]=a,e[6]=h,e[7]=0,e[8]=l,e[9]=d,e[10]=f,e[11]=0,e[12]=-(n*u+s*v+l*g),e[13]=-(c*u+a*v+d*g),e[14]=-(o*u+h*v+f*g),e[15]=1,e)}function U(){let e=new T(3);return T!=Float32Array&&(e[0]=0,e[1]=0,e[2]=0),e}function be(e){var t=new T(3);return t[0]=e[0],t[1]=e[1],t[2]=e[2],t}function Me(e){let t=e[0],r=e[1],i=e[2];return Math.sqrt(t*t+r*r+i*i)}function L(e,t,r){let i=new T(3);return i[0]=e,i[1]=t,i[2]=r,i}function $(e,t,r){return e[0]=t[0]-r[0],e[1]=t[1]-r[1],e[2]=t[2]-r[2],e}function ee(e,t){let r=t[0],i=t[1],n=t[2],s=r*r+i*i+n*n;return s>0&&(s=1/Math.sqrt(s)),e[0]=t[0]*s,e[1]=t[1]*s,e[2]=t[2]*s,e}function we(e,t){return e[0]*t[0]+e[1]*t[1]+e[2]*t[2]}function O(e,t,r){let i=t[0],n=t[1],s=t[2],l=r[0],c=r[1],a=r[2];return e[0]=n*a-s*c,e[1]=s*l-i*a,e[2]=i*c-n*l,e}function V(e,t,r){let i=t[0],n=t[1],s=t[2],l=r[3]*i+r[7]*n+r[11]*s+r[15];return l=l||1,e[0]=(r[0]*i+r[4]*n+r[8]*s+r[12])/l,e[1]=(r[1]*i+r[5]*n+r[9]*s+r[13])/l,e[2]=(r[2]*i+r[6]*n+r[10]*s+r[14])/l,e}const ye=Me;(function(){let e=U();return function(t,r,i,n,s,l){let c,a;for(r||(r=3),i||(i=0),n?a=Math.min(n*r+i,t.length):a=t.length,c=i;c<a;c+=r)e[0]=t[c],e[1]=t[c+1],e[2]=t[c+2],s(e,e,l),t[c]=e[0],t[c+1]=e[1],t[c+2]=e[2];return t}})();function Ee(){let e=new T(4);return T!=Float32Array&&(e[0]=0,e[1]=0,e[2]=0,e[3]=0),e}function Te(e,t){let r=t[0],i=t[1],n=t[2],s=t[3],l=r*r+i*i+n*n+s*s;return l>0&&(l=1/Math.sqrt(l)),e[0]=r*l,e[1]=i*l,e[2]=n*l,e[3]=s*l,e}(function(){let e=Ee();return function(t,r,i,n,s,l){let c,a;for(r||(r=4),i||(i=0),n?a=Math.min(n*r+i,t.length):a=t.length,c=i;c<a;c+=r)e[0]=t[c],e[1]=t[c+1],e[2]=t[c+2],e[3]=t[c+3],s(e,e,l),t[c]=e[0],t[c+1]=e[1],t[c+2]=e[2],t[c+3]=e[3];return t}})();function H(){let e=new T(4);return T!=Float32Array&&(e[0]=0,e[1]=0,e[2]=0),e[3]=1,e}function Pe(e,t,r){r=r*.5;let i=Math.sin(r);return e[0]=i*t[0],e[1]=i*t[1],e[2]=i*t[2],e[3]=Math.cos(r),e}function I(e,t,r,i){let n=t[0],s=t[1],l=t[2],c=t[3],a=r[0],d=r[1],o=r[2],h=r[3],f,p,u,v,g;return p=n*a+s*d+l*o+c*h,p<0&&(p=-p,a=-a,d=-d,o=-o,h=-h),1-p>C?(f=Math.acos(p),u=Math.sin(f),v=Math.sin((1-i)*f)/u,g=Math.sin(i*f)/u):(v=1-i,g=i),e[0]=v*n+g*a,e[1]=v*s+g*d,e[2]=v*l+g*o,e[3]=v*c+g*h,e}function Se(e,t){let r=t[0]+t[4]+t[8],i;if(r>0)i=Math.sqrt(r+1),e[3]=.5*i,i=.5/i,e[0]=(t[5]-t[7])*i,e[1]=(t[6]-t[2])*i,e[2]=(t[1]-t[3])*i;else{let n=0;t[4]>t[0]&&(n=1),t[8]>t[n*3+n]&&(n=2);let s=(n+1)%3,l=(n+2)%3;i=Math.sqrt(t[n*3+n]-t[s*3+s]-t[l*3+l]+1),e[n]=.5*i,i=.5/i,e[3]=(t[s*3+l]-t[l*3+s])*i,e[s]=(t[s*3+n]+t[n*3+s])*i,e[l]=(t[l*3+n]+t[n*3+l])*i}return e}const te=Te;(function(){let e=U(),t=L(1,0,0),r=L(0,1,0);return function(i,n,s){let l=we(n,s);return l<-.999999?(O(e,t,n),ye(e)<1e-6&&O(e,r,n),ee(e,e),Pe(i,e,Math.PI),i):l>.999999?(i[0]=0,i[1]=0,i[2]=0,i[3]=1,i):(O(e,n,s),i[0]=e[0],i[1]=e[1],i[2]=e[2],i[3]=1+l,te(i,i))}})();(function(){let e=H(),t=H();return function(r,i,n,s,l,c){return I(e,i,l,c),I(t,n,s,c),I(r,e,t,2*c*(1-c)),r}})();(function(){let e=oe();return function(t,r,i,n){return e[0]=i[0],e[3]=i[1],e[6]=i[2],e[1]=n[0],e[4]=n[1],e[7]=n[2],e[2]=-r[0],e[5]=-r[1],e[8]=-r[2],te(t,Se(t,e))}})();function q(){let e=new T(2);return T!=Float32Array&&(e[0]=0,e[1]=0),e}function W(e,t){let r=new T(2);return r[0]=e,r[1]=t,r}function Ne(e,t,r){return e[0]=t[0]+r[0],e[1]=t[1]+r[1],e}function k(e,t,r){return e[0]=t[0]-r[0],e[1]=t[1]-r[1],e}(function(){let e=q();return function(t,r,i,n,s,l){let c,a;for(r||(r=2),i||(i=0),n?a=Math.min(n*r+i,t.length):a=t.length,c=i;c<a;c+=r)e[0]=t[c],e[1]=t[c+1],s(e,e,l),t[c]=e[0],t[c+1]=e[1];return t}})();class Ge{constructor(t,r){m(this,"_label");m(this,"_device");m(this,"_vertices");m(this,"_indices");m(this,"_vertexBuffer");m(this,"_indexBuffer");this._label=r,this._device=t,this._vertices=[],this._indices=[]}initialize(){const t=[];for(let n=0;n<this._vertices.length;n++){const{position:s,normal:l,tangent:c,texCoord:a}=this._vertices[n];t.push(...s,...l,...c,...a)}const r=new Float32Array(t);this._vertexBuffer=this._device.createBuffer({label:`${this._label}-vertex-buffer`,size:r.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),this._device.queue.writeBuffer(this._vertexBuffer,0,r);const i=new Uint32Array(this._indices);this._indexBuffer=this._device.createBuffer({label:`${this._label}-index-buffer`,size:i.byteLength,usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST}),this._device.queue.writeBuffer(this._indexBuffer,0,i)}draw(t){if(!this._vertexBuffer||!this._indexBuffer){console.error(`${this._label} mesh was not initialized!`);return}if(!t){console.error(`GPURenderPassEncoder was not passed to ${this._label} mesh`);return}t.setVertexBuffer(0,this._vertexBuffer),t.setIndexBuffer(this._indexBuffer,"uint32"),t.drawIndexed(this._indices.length)}}class Ue extends Ge{constructor(t,r){super(t,"sphere");const i=500,n=500,s=-2*Math.PI/i,l=-1*Math.PI/n;for(let c=0;c<=n;c++){const a=R();X(a,R(),l*c,L(0,0,1));const d=U();V(d,L(0,-r,0),a);for(let o=0;o<=i;o++){const h=R();X(h,R(),s*o,L(0,1,0));const f=U();V(f,d,h);const p=U();ee(p,f);const u=W(1-o/i,1-c/n);this._vertices.push({position:f,normal:p,texCoord:u,tangent:U()})}}for(let c=0;c<n;c++){const a=(i+1)*c;for(let d=0;d<i;d++){const o=a+d,h=a+d+1,f=a+(d+1)%(i+1)+(i+1),p=f-1;this._indices.push(o),this._indices.push(f),this._indices.push(h),this._indices.push(o),this._indices.push(p),this._indices.push(f);const u=this._vertices[o].position,v=this._vertices[h].position,g=this._vertices[f].position,x=this._vertices[p].position,b=this.calculateTangent(u,v,g,this._vertices[o].texCoord,this._vertices[h].texCoord,this._vertices[f].texCoord),M=this.calculateTangent(u,g,x,this._vertices[o].texCoord,this._vertices[f].texCoord,this._vertices[p].texCoord);this._vertices[o].tangent=b,this._vertices[h].tangent=b,this._vertices[f].tangent=b,this._vertices[o].tangent=M,this._vertices[f].tangent=M,this._vertices[p].tangent=M}}this.initialize()}calculateTangent(t,r,i,n,s,l){const c=U();$(c,r,t);const a=U();$(a,i,t);const d=q();k(d,s,n);const o=q();k(o,l,n);const h=1/(d[0]*o[1]-o[0]*d[1]),f=U();return f[0]=h*(o[1]*c[0]-d[1]*a[0]),f[1]=h*(o[1]*c[1]-d[1]*a[1]),f[2]=h*(o[1]*c[2]-d[1]*a[2]),f}}class ze{constructor({position:t,center:r,up:i}){m(this,"_position");m(this,"_center");m(this,"_up");m(this,"_rotate");m(this,"_isMobile");m(this,"_isDragging");m(this,"_initialX");m(this,"_initialY");this._position=t,this._center=r,this._up=i,this._rotate=W(0,0),this._isDragging=!1,this._initialX=0,this._initialY=0,this._isMobile=/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent),this.initializeEvent()}get position(){const t=this.getViewRotationMatrix(),r=U();return V(r,this._position,t),r}getViewMatrix(){const t=R(),r=this.getViewRotationMatrix(),i=U(),n=U(),s=U();return V(i,this._position,r),V(n,this._center,r),V(s,this._up,r),xe(t,i,n,s),t}initializeEvent(){const t=this._isMobile?"touchstart":"mousedown",r=this._isMobile?"touchmove":"mousemove",i=this._isMobile?"touchend":"mouseup";document.addEventListener(t,n=>{this._isDragging=!0,this._initialX=this._isMobile?n.touches[0].clientX:n.clientX,this._initialY=this._isMobile?n.touches[0].clientY:n.clientY}),document.addEventListener(r,n=>{if(this._isDragging){const s=this._isMobile?n.touches[0].clientX:n.clientX,l=this._isMobile?n.touches[0].clientY:n.clientY,c=s-this._initialX,a=l-this._initialY;this._rotate=Ne(this._rotate,this._rotate,W(a/10,c/10)),this._initialX=s,this._initialY=l,n.preventDefault()}}),document.addEventListener(i,()=>{this._isDragging=!1})}getViewRotationMatrix(){const t=R();return Q(t,t,Y(this._rotate[1])),J(t,t,Y(this._rotate[0])),t}}class Fe{static getModelMatrix({translation:t,scaling:r,rotation:i}){const n=R();return ge(n,n,t),ue(n,n,r),J(n,n,i[0]),ve(n,n,i[1]),Q(n,n,i[2]),n}}class j{static async loadImageBitmap(t){const i=await(await fetch(t)).blob();return await createImageBitmap(i,{colorSpaceConversion:"none"})}static async generateMips(t,r){let i=t;const n=[i];let s=0;for(;s<r&&(i.width>1||i.height>1);)i=await this.createNextMipLevelRgba8Unorm(i),n.push(i),s++;return n}static async createNextMipLevelRgba8Unorm(t){const r=Math.max(1,t.width/2|0),i=Math.max(1,t.height/2|0),n=document.createElement("canvas");n.width=r,n.height=i;const s=n.getContext("2d");if(!s)throw new Error("Unable to get 2D context");return s.drawImage(t,0,0,r,i),createImageBitmap(n)}}class A{constructor(t){m(this,"_device");m(this,"_texture");this._device=t}get view(){return this._texture||console.error("You need to initialize texture first!"),this._texture.createView()}async initialize(t,r=0){const i=await j.loadImageBitmap(t),n=await j.generateMips(i,r);if(this._texture=this._device.createTexture({label:"yellow F on red",size:[n[0].width,n[0].height],mipLevelCount:n.length,format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),!this._texture){console.error("Failed to load texture");return}n.forEach((s,l)=>{this._device.queue.copyExternalImageToTexture({source:s,flipY:!1},{texture:this._texture,mipLevel:l},{width:s.width,height:s.height})})}}const K=document.documentElement.clientWidth,Z=document.documentElement.clientHeight;async function Be(){const e=new le(document.querySelector("canvas"));if(await e.initialize(K,Z),!e.device)return;const t="",r=new A(e.device);await r.initialize(t+"pbr/antique-grate1-albedo.jpg");const i=new A(e.device);await i.initialize(t+"pbr/antique-grate1-normal-dx.jpg");const n=new A(e.device);await n.initialize(t+"pbr/antique-grate1-metallic.jpg");const s=new A(e.device);await s.initialize(t+"pbr/antique-grate1-roughness.jpg");const l=new A(e.device);await l.initialize(t+"pbr/antique-grate1-ao.jpg");const c=new A(e.device);await c.initialize(t+"cubemap/air_museum_playground_brdf.jpg");const a=e.device.createSampler({magFilter:"linear",minFilter:"linear"}),d=new Ue(e.device,1),o=new ce({label:"main",device:e.device,vertexShader:re,fragmentShader:se,bindGroupLayouts:[e.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:3,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float",viewDimension:"2d"}},{binding:4,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float",viewDimension:"2d"}},{binding:5,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float",viewDimension:"2d"}},{binding:6,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float",viewDimension:"2d"}},{binding:7,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float",viewDimension:"2d"}},{binding:8,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float",viewDimension:"2d"}}]})]}),h=e.device.createBuffer({label:"vertex uniforms",size:64*Float32Array.BYTES_PER_ELEMENT,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),f=e.device.createBuffer({label:"fragment uniforms",size:4*Float32Array.BYTES_PER_ELEMENT,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),p=e.device.createBindGroup({label:"bind group for object",layout:o.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:h}},{binding:1,resource:{buffer:f}},{binding:2,resource:a},{binding:3,resource:r.view},{binding:4,resource:i.view},{binding:5,resource:n.view},{binding:6,resource:s.view},{binding:7,resource:l.view},{binding:8,resource:c.view}]}),u=L(0,0,0),v=L(.7,.7,.7),g=L(0,0,0),x=new ze({position:L(0,0,2.5),center:L(0,0,0),up:L(0,1,0)}),b=R();_e(b,Y(45),K/Z,.1,100);function M(){var S,N;const E=Fe.getModelMatrix({translation:u,scaling:v,rotation:g}),P=x.getViewMatrix(),w=fe(E);pe(w,w),he(w,w),(S=e.device)==null||S.queue.writeBuffer(h,0,new Float32Array([...E,...P,...b,...w]));const G=be(x.position);(N=e.device)==null||N.queue.writeBuffer(f,0,new Float32Array([...G,0]));const y=e.getPass();o.use(y),y==null||y.setBindGroup(0,p),d.draw(y),e.endPass(),requestAnimationFrame(M)}M()}Be();
