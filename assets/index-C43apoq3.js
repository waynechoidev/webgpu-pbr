var le=Object.defineProperty;var ce=(e,t,r)=>t in e?le(e,t,{enumerable:!0,configurable:!0,writable:!0,value:r}):e[t]=r;var _=(e,t,r)=>(ce(e,typeof t!="symbol"?t+"":t,r),r);(function(){const t=document.createElement("link").relList;if(t&&t.supports&&t.supports("modulepreload"))return;for(const n of document.querySelectorAll('link[rel="modulepreload"]'))i(n);new MutationObserver(n=>{for(const s of n)if(s.type==="childList")for(const a of s.addedNodes)a.tagName==="LINK"&&a.rel==="modulepreload"&&i(a)}).observe(document,{childList:!0,subtree:!0});function r(n){const s={};return n.integrity&&(s.integrity=n.integrity),n.referrerPolicy&&(s.referrerPolicy=n.referrerPolicy),n.crossOrigin==="use-credentials"?s.credentials="include":n.crossOrigin==="anonymous"?s.credentials="omit":s.credentials="same-origin",s}function i(n){if(n.ep)return;n.ep=!0;const s=r(n);fetch(n.href,s)}})();var de=`struct VSOutput {
  @builtin(position) position: vec4f,
  @location(0) posWorld: vec3f,
  @location(1) normalWorld: vec3f,
  @location(2) tangentWorld: vec3f,
  @location(3) texCoord: vec2f,
};

@group(0) @binding(2) var mySampler: sampler;

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
@group(0) @binding(3) var heightMap: texture_2d<f32>;

@vertex fn vs(
  input: Vertex,
) -> VSOutput {
  var output: VSOutput;
  
  output.normalWorld = normalize(uni.invTransposedModel * vec4f(input.norm, 1.0)).xyz;
  output.tangentWorld = normalize(uni.model * vec4f(input.tangent, 1.0)).xyz;
  output.texCoord = input.tex;

  let heightScale:f32 = 0.1;
  let height:f32 = textureSampleLevel(heightMap, mySampler, input.tex, 0).x;
  var newPos: vec3f = input.pos + (output.normalWorld * height * heightScale);

	output.posWorld = (uni.model * vec4f(newPos, 1.0)).xyz;
  output.position = uni.projection * uni.view * uni.model * vec4f(newPos , 1.0);
  return output;
}`,fe=`struct VSOutput {
  @builtin(position) position: vec4f,
  @location(0) posWorld: vec3f,
  @location(1) normalWorld: vec3f,
  @location(2) tangentWorld: vec3f,
  @location(3) texCoord: vec2f,
};

@group(0) @binding(2) var mySampler: sampler;
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

@group(0) @binding(4) var albedoMap: texture_2d<f32>;
@group(0) @binding(5) var normalMap: texture_2d<f32>;
@group(0) @binding(6) var metallicMap: texture_2d<f32>;
@group(0) @binding(7) var roughnessMap: texture_2d<f32>;
@group(0) @binding(8) var aoMap: texture_2d<f32>;
@group(0) @binding(9) var brdfLUT: texture_2d<f32>;

@group(0) @binding(10) var envCubemap: texture_cube<f32>;
@group(0) @binding(11) var irradianceCubemap: texture_cube<f32>;

@fragment fn fs(input: VSOutput) -> @location(0) vec4f {
  
  let albedo:vec3f = pow(textureSample(albedoMap, mySampler, input.texCoord).rgb, vec3(2.2));
  let metallic:f32 = textureSample(metallicMap, mySampler, input.texCoord).r;
  let roughness:f32 = textureSample(roughnessMap, mySampler, input.texCoord).r;
  let ao:f32 = textureSample(aoMap, mySampler, input.texCoord).r;

  let normalWorld:vec3f = normalize(input.normalWorld);
  
  
  let tangent:vec3f = normalize(input.tangentWorld - dot(input.tangentWorld, normalWorld) * normalWorld);
  let bitangent:vec3f = cross(normalWorld, tangent);
  let TBN:mat3x3f = mat3x3f(tangent, bitangent, normalWorld);
  let N:vec3f = normalize(TBN * (textureSample(normalMap, mySampler, input.texCoord).xyz * 2.0 - 1.0));

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

  var F:vec3f = fresnelSchlick(max(dot(H, V), 0.0), F0);
  let G:f32 = GeometrySmith(N, V, L, roughness);
  let NDF:f32 = DistributionGGX(N, H, roughness);

  let numerator:vec3f = NDF * G * F;
  
  let denominator:f32 = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
  var specular:vec3f = numerator / denominator;

  var kS:vec3f = F;

  var kD:vec3f = (vec3(1.0) - kS) * (1.0 - metallic);

  
  let NdotL:f32 = max(dot(N, L), 0.0);

  let directLight:vec3f = (kD * albedo / PI + specular) * radiance * NdotL;

  
  F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);

  kS = F;
  kD = 1.0 - kS;
  kD *= 1.0 - metallic;

  
  let irradiance:vec3f = pow(textureSample(irradianceCubemap, mySampler, N).rgb, vec3(2.2));
  let diffuse:vec3f = irradiance * albedo;

  
  const MAX_REFLECTION_LOD:f32 = 4.0;
  
  let prefilteredColor:vec3f = pow(textureSampleLevel(envCubemap, mySampler, R,  roughness * MAX_REFLECTION_LOD).rgb, vec3(2.2));
  let brdf:vec2f  = textureSample(brdfLUT, mySampler, vec2(max(dot(N, V), 0.0), roughness)).rg;
  specular = prefilteredColor * (F * brdf.x + brdf.y);

  let ambient:vec3f = (kD * diffuse + specular) * ao;

  var color:vec3f = directLight + ambient;

  
  color = color / (color + vec3(1.0));
  
  color = pow(color, vec3(1.0/2.2));

  return vec4f(color, 1.0);
}`,ue=`struct VSOutput {
  @builtin(position) position: vec4f,
  @location(0) posWorld: vec3f
};

struct Vertex {
  @location(0) pos: vec3f,
  @location(1) norm: vec3f,
  @location(2) tangent: vec3f,
  @location(3) tex: vec2f,
};

struct Uniforms {
  view: mat4x4f,
  projection: mat4x4f
};

@group(0) @binding(0) var<uniform> uni: Uniforms;

@vertex fn vs(
  input: Vertex,
) -> VSOutput {
  let rotView: mat4x4<f32> = mat4x4<f32>(
    vec4f(uni.view[0].xyz, 0.0),
    vec4f(uni.view[1].xyz, 0.0), 
    vec4f(uni.view[2].xyz, 0.0),
    vec4f(0.0, 0.0, 0.0, 1.0)
  );
  let clipPos:vec4f = uni.projection * uni.view * vec4f(input.pos * 20.0, 1.0);

  var output: VSOutput;
  output.posWorld = input.pos;
  output.position = clipPos.xyww;
  return output;
}`,pe=`struct VSOutput {
  @builtin(position) position: vec4f,
  @location(0) posWorld: vec3f
};

@group(0) @binding(1) var mySampler: sampler;
@group(0) @binding(2) var envCubemap: texture_cube<f32>;

@fragment fn fs(input: VSOutput) -> @location(0) vec4f {
  let envColor:vec3f = textureSample(envCubemap, mySampler, input.posWorld).rgb;
  
  return vec4f(envColor, 1.0);
}`;class he{constructor(t){_(this,"_canvas");_(this,"_adapter");_(this,"_device");_(this,"_context");_(this,"_encoder");_(this,"_pass");_(this,"_depthTexture");this._canvas=t}get device(){return this._device}async initialize(t,r){var i,n,s;if(this._canvas.width=t,this._canvas.height=r,this._adapter=await((i=navigator.gpu)==null?void 0:i.requestAdapter()),this._device=await((n=this._adapter)==null?void 0:n.requestDevice()),!this._device){const a="Your device does not support WebGPU.";console.error(a);const o=document.createElement("p");o.innerHTML=a,(s=document.querySelector("body"))==null||s.prepend(o)}this._context=this._canvas.getContext("webgpu"),this._context.configure({device:this._device,format:navigator.gpu.getPreferredCanvasFormat()})}getPass(){if(!this._context||!this._device){console.error("RenderEnv was not initialized!");return}const t=this._context.getCurrentTexture();(!this._depthTexture||this._depthTexture.width!==t.width||this._depthTexture.height!==t.height)&&(this._depthTexture&&this._depthTexture.destroy(),this._depthTexture=this._device.createTexture({size:[t.width,t.height],format:"depth24plus",usage:GPUTextureUsage.RENDER_ATTACHMENT}));const r={colorAttachments:[{view:t.createView(),clearValue:[0,0,0,1],loadOp:"clear",storeOp:"store"}],depthStencilAttachment:{view:this._depthTexture.createView(),depthClearValue:1,depthLoadOp:"clear",depthStoreOp:"store"}};return this._encoder=this._device.createCommandEncoder(),this._pass=this._encoder.beginRenderPass(r),this._pass}endPass(){var r,i;if(!this._device){console.error("RenderEnv was not initialized!");return}(r=this._pass)==null||r.end();const t=(i=this._encoder)==null?void 0:i.finish();t&&this._device.queue.submit([t])}}class K{constructor({label:t,device:r,vertexShader:i,fragmentShader:n,buffer:s,bindGroupLayouts:a}){_(this,"_label");_(this,"_pipeline");this._label=t,this._pipeline=r.createRenderPipeline({label:`${t} pipeline`,layout:a?r.createPipelineLayout({bindGroupLayouts:a}):"auto",vertex:{module:r.createShaderModule({label:`${t} vertex shader`,code:i}),buffers:s??[{arrayStride:11*Float32Array.BYTES_PER_ELEMENT,attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:3*Float32Array.BYTES_PER_ELEMENT,format:"float32x3"},{shaderLocation:2,offset:6*Float32Array.BYTES_PER_ELEMENT,format:"float32x3"},{shaderLocation:3,offset:9*Float32Array.BYTES_PER_ELEMENT,format:"float32x2"}]}]},fragment:{module:r.createShaderModule({label:`${t} fragment shader`,code:n}),targets:[{format:navigator.gpu.getPreferredCanvasFormat()}]},primitive:{topology:"triangle-list",cullMode:"back"},depthStencil:{depthWriteEnabled:!0,depthCompare:"less-equal",format:"depth24plus"}})}getBindGroupLayout(t){return this._pipeline.getBindGroupLayout(t)}use(t){if(!t){console.error(`GPURenderPassEncoder was not passed to ${this._label} pipeline`);return}t.setPipeline(this._pipeline)}}const O=1e-6;let T=typeof Float32Array<"u"?Float32Array:Array;const ge=Math.PI/180;function j(e){return e*ge}function ve(){let e=new T(9);return T!=Float32Array&&(e[1]=0,e[2]=0,e[3]=0,e[5]=0,e[6]=0,e[7]=0),e[0]=1,e[4]=1,e[8]=1,e}function V(){let e=new T(16);return T!=Float32Array&&(e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[11]=0,e[12]=0,e[13]=0,e[14]=0),e[0]=1,e[5]=1,e[10]=1,e[15]=1,e}function me(e){let t=new T(16);return t[0]=e[0],t[1]=e[1],t[2]=e[2],t[3]=e[3],t[4]=e[4],t[5]=e[5],t[6]=e[6],t[7]=e[7],t[8]=e[8],t[9]=e[9],t[10]=e[10],t[11]=e[11],t[12]=e[12],t[13]=e[13],t[14]=e[14],t[15]=e[15],t}function _e(e){return e[0]=1,e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=1,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[10]=1,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,e}function be(e,t){if(e===t){let r=t[1],i=t[2],n=t[3],s=t[6],a=t[7],o=t[11];e[1]=t[4],e[2]=t[8],e[3]=t[12],e[4]=r,e[6]=t[9],e[7]=t[13],e[8]=i,e[9]=s,e[11]=t[14],e[12]=n,e[13]=a,e[14]=o}else e[0]=t[0],e[1]=t[4],e[2]=t[8],e[3]=t[12],e[4]=t[1],e[5]=t[5],e[6]=t[9],e[7]=t[13],e[8]=t[2],e[9]=t[6],e[10]=t[10],e[11]=t[14],e[12]=t[3],e[13]=t[7],e[14]=t[11],e[15]=t[15];return e}function xe(e,t){let r=t[0],i=t[1],n=t[2],s=t[3],a=t[4],o=t[5],l=t[6],f=t[7],c=t[8],u=t[9],d=t[10],p=t[11],v=t[12],m=t[13],g=t[14],x=t[15],y=r*o-i*a,w=r*l-n*a,E=r*f-s*a,P=i*l-n*o,G=i*f-s*o,R=n*f-s*l,N=c*m-u*v,U=c*g-d*v,z=c*x-p*v,F=u*g-d*m,B=u*x-p*m,M=d*x-p*g,b=y*M-w*B+E*F+P*z-G*U+R*N;return b?(b=1/b,e[0]=(o*M-l*B+f*F)*b,e[1]=(n*B-i*M-s*F)*b,e[2]=(m*R-g*G+x*P)*b,e[3]=(d*G-u*R-p*P)*b,e[4]=(l*z-a*M-f*U)*b,e[5]=(r*M-n*z+s*U)*b,e[6]=(g*E-v*R-x*w)*b,e[7]=(c*R-d*E+p*w)*b,e[8]=(a*B-o*z+f*N)*b,e[9]=(i*z-r*B-s*N)*b,e[10]=(v*G-m*E+x*y)*b,e[11]=(u*E-c*G-p*y)*b,e[12]=(o*U-a*F-l*N)*b,e[13]=(r*F-i*U+n*N)*b,e[14]=(m*w-v*P-g*y)*b,e[15]=(c*P-u*w+d*y)*b,e):null}function ye(e,t,r){let i=r[0],n=r[1],s=r[2],a,o,l,f,c,u,d,p,v,m,g,x;return t===e?(e[12]=t[0]*i+t[4]*n+t[8]*s+t[12],e[13]=t[1]*i+t[5]*n+t[9]*s+t[13],e[14]=t[2]*i+t[6]*n+t[10]*s+t[14],e[15]=t[3]*i+t[7]*n+t[11]*s+t[15]):(a=t[0],o=t[1],l=t[2],f=t[3],c=t[4],u=t[5],d=t[6],p=t[7],v=t[8],m=t[9],g=t[10],x=t[11],e[0]=a,e[1]=o,e[2]=l,e[3]=f,e[4]=c,e[5]=u,e[6]=d,e[7]=p,e[8]=v,e[9]=m,e[10]=g,e[11]=x,e[12]=a*i+c*n+v*s+t[12],e[13]=o*i+u*n+m*s+t[13],e[14]=l*i+d*n+g*s+t[14],e[15]=f*i+p*n+x*s+t[15]),e}function we(e,t,r){let i=r[0],n=r[1],s=r[2];return e[0]=t[0]*i,e[1]=t[1]*i,e[2]=t[2]*i,e[3]=t[3]*i,e[4]=t[4]*n,e[5]=t[5]*n,e[6]=t[6]*n,e[7]=t[7]*n,e[8]=t[8]*s,e[9]=t[9]*s,e[10]=t[10]*s,e[11]=t[11]*s,e[12]=t[12],e[13]=t[13],e[14]=t[14],e[15]=t[15],e}function Z(e,t,r,i){let n=i[0],s=i[1],a=i[2],o=Math.sqrt(n*n+s*s+a*a),l,f,c,u,d,p,v,m,g,x,y,w,E,P,G,R,N,U,z,F,B,M,b,S;return o<O?null:(o=1/o,n*=o,s*=o,a*=o,l=Math.sin(r),f=Math.cos(r),c=1-f,u=t[0],d=t[1],p=t[2],v=t[3],m=t[4],g=t[5],x=t[6],y=t[7],w=t[8],E=t[9],P=t[10],G=t[11],R=n*n*c+f,N=s*n*c+a*l,U=a*n*c-s*l,z=n*s*c-a*l,F=s*s*c+f,B=a*s*c+n*l,M=n*a*c+s*l,b=s*a*c-n*l,S=a*a*c+f,e[0]=u*R+m*N+w*U,e[1]=d*R+g*N+E*U,e[2]=p*R+x*N+P*U,e[3]=v*R+y*N+G*U,e[4]=u*z+m*F+w*B,e[5]=d*z+g*F+E*B,e[6]=p*z+x*F+P*B,e[7]=v*z+y*F+G*B,e[8]=u*M+m*b+w*S,e[9]=d*M+g*b+E*S,e[10]=p*M+x*b+P*S,e[11]=v*M+y*b+G*S,t!==e&&(e[12]=t[12],e[13]=t[13],e[14]=t[14],e[15]=t[15]),e)}function ie(e,t,r){let i=Math.sin(r),n=Math.cos(r),s=t[4],a=t[5],o=t[6],l=t[7],f=t[8],c=t[9],u=t[10],d=t[11];return t!==e&&(e[0]=t[0],e[1]=t[1],e[2]=t[2],e[3]=t[3],e[12]=t[12],e[13]=t[13],e[14]=t[14],e[15]=t[15]),e[4]=s*n+f*i,e[5]=a*n+c*i,e[6]=o*n+u*i,e[7]=l*n+d*i,e[8]=f*n-s*i,e[9]=c*n-a*i,e[10]=u*n-o*i,e[11]=d*n-l*i,e}function re(e,t,r){let i=Math.sin(r),n=Math.cos(r),s=t[0],a=t[1],o=t[2],l=t[3],f=t[8],c=t[9],u=t[10],d=t[11];return t!==e&&(e[4]=t[4],e[5]=t[5],e[6]=t[6],e[7]=t[7],e[12]=t[12],e[13]=t[13],e[14]=t[14],e[15]=t[15]),e[0]=s*n-f*i,e[1]=a*n-c*i,e[2]=o*n-u*i,e[3]=l*n-d*i,e[8]=s*i+f*n,e[9]=a*i+c*n,e[10]=o*i+u*n,e[11]=l*i+d*n,e}function Me(e,t,r){let i=Math.sin(r),n=Math.cos(r),s=t[0],a=t[1],o=t[2],l=t[3],f=t[4],c=t[5],u=t[6],d=t[7];return t!==e&&(e[8]=t[8],e[9]=t[9],e[10]=t[10],e[11]=t[11],e[12]=t[12],e[13]=t[13],e[14]=t[14],e[15]=t[15]),e[0]=s*n+f*i,e[1]=a*n+c*i,e[2]=o*n+u*i,e[3]=l*n+d*i,e[4]=f*n-s*i,e[5]=c*n-a*i,e[6]=u*n-o*i,e[7]=d*n-l*i,e}function Se(e,t,r,i,n){const s=1/Math.tan(t/2);if(e[0]=s/r,e[1]=0,e[2]=0,e[3]=0,e[4]=0,e[5]=s,e[6]=0,e[7]=0,e[8]=0,e[9]=0,e[11]=-1,e[12]=0,e[13]=0,e[15]=0,n!=null&&n!==1/0){const a=1/(i-n);e[10]=(n+i)*a,e[14]=2*n*i*a}else e[10]=-1,e[14]=-2*i;return e}const Te=Se;function Ee(e,t,r,i){let n,s,a,o,l,f,c,u,d,p,v=t[0],m=t[1],g=t[2],x=i[0],y=i[1],w=i[2],E=r[0],P=r[1],G=r[2];return Math.abs(v-E)<O&&Math.abs(m-P)<O&&Math.abs(g-G)<O?_e(e):(c=v-E,u=m-P,d=g-G,p=1/Math.sqrt(c*c+u*u+d*d),c*=p,u*=p,d*=p,n=y*d-w*u,s=w*c-x*d,a=x*u-y*c,p=Math.sqrt(n*n+s*s+a*a),p?(p=1/p,n*=p,s*=p,a*=p):(n=0,s=0,a=0),o=u*a-d*s,l=d*n-c*a,f=c*s-u*n,p=Math.sqrt(o*o+l*l+f*f),p?(p=1/p,o*=p,l*=p,f*=p):(o=0,l=0,f=0),e[0]=n,e[1]=o,e[2]=c,e[3]=0,e[4]=s,e[5]=l,e[6]=u,e[7]=0,e[8]=a,e[9]=f,e[10]=d,e[11]=0,e[12]=-(n*v+s*m+a*g),e[13]=-(o*v+l*m+f*g),e[14]=-(c*v+u*m+d*g),e[15]=1,e)}function C(){let e=new T(3);return T!=Float32Array&&(e[0]=0,e[1]=0,e[2]=0),e}function Pe(e){var t=new T(3);return t[0]=e[0],t[1]=e[1],t[2]=e[2],t}function Ge(e){let t=e[0],r=e[1],i=e[2];return Math.sqrt(t*t+r*r+i*i)}function h(e,t,r){let i=new T(3);return i[0]=e,i[1]=t,i[2]=r,i}function J(e,t,r){return e[0]=t[0]-r[0],e[1]=t[1]-r[1],e[2]=t[2]-r[2],e}function se(e,t){let r=t[0],i=t[1],n=t[2],s=r*r+i*i+n*n;return s>0&&(s=1/Math.sqrt(s)),e[0]=t[0]*s,e[1]=t[1]*s,e[2]=t[2]*s,e}function Ue(e,t){return e[0]*t[0]+e[1]*t[1]+e[2]*t[2]}function Y(e,t,r){let i=t[0],n=t[1],s=t[2],a=r[0],o=r[1],l=r[2];return e[0]=n*l-s*o,e[1]=s*a-i*l,e[2]=i*o-n*a,e}function D(e,t,r){let i=t[0],n=t[1],s=t[2],a=r[3]*i+r[7]*n+r[11]*s+r[15];return a=a||1,e[0]=(r[0]*i+r[4]*n+r[8]*s+r[12])/a,e[1]=(r[1]*i+r[5]*n+r[9]*s+r[13])/a,e[2]=(r[2]*i+r[6]*n+r[10]*s+r[14])/a,e}const Ne=Ge;(function(){let e=C();return function(t,r,i,n,s,a){let o,l;for(r||(r=3),i||(i=0),n?l=Math.min(n*r+i,t.length):l=t.length,o=i;o<l;o+=r)e[0]=t[o],e[1]=t[o+1],e[2]=t[o+2],s(e,e,a),t[o]=e[0],t[o+1]=e[1],t[o+2]=e[2];return t}})();function ze(){let e=new T(4);return T!=Float32Array&&(e[0]=0,e[1]=0,e[2]=0,e[3]=0),e}function Fe(e,t){let r=t[0],i=t[1],n=t[2],s=t[3],a=r*r+i*i+n*n+s*s;return a>0&&(a=1/Math.sqrt(a)),e[0]=r*a,e[1]=i*a,e[2]=n*a,e[3]=s*a,e}(function(){let e=ze();return function(t,r,i,n,s,a){let o,l;for(r||(r=4),i||(i=0),n?l=Math.min(n*r+i,t.length):l=t.length,o=i;o<l;o+=r)e[0]=t[o],e[1]=t[o+1],e[2]=t[o+2],e[3]=t[o+3],s(e,e,a),t[o]=e[0],t[o+1]=e[1],t[o+2]=e[2],t[o+3]=e[3];return t}})();function Q(){let e=new T(4);return T!=Float32Array&&(e[0]=0,e[1]=0,e[2]=0),e[3]=1,e}function Be(e,t,r){r=r*.5;let i=Math.sin(r);return e[0]=i*t[0],e[1]=i*t[1],e[2]=i*t[2],e[3]=Math.cos(r),e}function q(e,t,r,i){let n=t[0],s=t[1],a=t[2],o=t[3],l=r[0],f=r[1],c=r[2],u=r[3],d,p,v,m,g;return p=n*l+s*f+a*c+o*u,p<0&&(p=-p,l=-l,f=-f,c=-c,u=-u),1-p>O?(d=Math.acos(p),v=Math.sin(d),m=Math.sin((1-i)*d)/v,g=Math.sin(i*d)/v):(m=1-i,g=i),e[0]=m*n+g*l,e[1]=m*s+g*f,e[2]=m*a+g*c,e[3]=m*o+g*u,e}function Re(e,t){let r=t[0]+t[4]+t[8],i;if(r>0)i=Math.sqrt(r+1),e[3]=.5*i,i=.5/i,e[0]=(t[5]-t[7])*i,e[1]=(t[6]-t[2])*i,e[2]=(t[1]-t[3])*i;else{let n=0;t[4]>t[0]&&(n=1),t[8]>t[n*3+n]&&(n=2);let s=(n+1)%3,a=(n+2)%3;i=Math.sqrt(t[n*3+n]-t[s*3+s]-t[a*3+a]+1),e[n]=.5*i,i=.5/i,e[3]=(t[s*3+a]-t[a*3+s])*i,e[s]=(t[s*3+n]+t[n*3+s])*i,e[a]=(t[a*3+n]+t[n*3+a])*i}return e}const ae=Fe;(function(){let e=C(),t=h(1,0,0),r=h(0,1,0);return function(i,n,s){let a=Ue(n,s);return a<-.999999?(Y(e,t,n),Ne(e)<1e-6&&Y(e,r,n),se(e,e),Be(i,e,Math.PI),i):a>.999999?(i[0]=0,i[1]=0,i[2]=0,i[3]=1,i):(Y(e,n,s),i[0]=e[0],i[1]=e[1],i[2]=e[2],i[3]=1+a,ae(i,i))}})();(function(){let e=Q(),t=Q();return function(r,i,n,s,a,o){return q(e,i,a,o),q(t,n,s,o),q(r,e,t,2*o*(1-o)),r}})();(function(){let e=ve();return function(t,r,i,n){return e[0]=i[0],e[3]=i[1],e[6]=i[2],e[1]=n[0],e[4]=n[1],e[7]=n[2],e[2]=-r[0],e[5]=-r[1],e[8]=-r[2],ae(t,Re(t,e))}})();function X(){let e=new T(2);return T!=Float32Array&&(e[0]=0,e[1]=0),e}function L(e,t){let r=new T(2);return r[0]=e,r[1]=t,r}function Ce(e,t,r){return e[0]=t[0]+r[0],e[1]=t[1]+r[1],e}function ee(e,t,r){return e[0]=t[0]-r[0],e[1]=t[1]-r[1],e}(function(){let e=X();return function(t,r,i,n,s,a){let o,l;for(r||(r=2),i||(i=0),n?l=Math.min(n*r+i,t.length):l=t.length,o=i;o<l;o+=r)e[0]=t[o],e[1]=t[o+1],s(e,e,a),t[o]=e[0],t[o+1]=e[1];return t}})();class oe{constructor(t,r){_(this,"_label");_(this,"_device");_(this,"_vertices");_(this,"_indices");_(this,"_vertexBuffer");_(this,"_indexBuffer");this._label=r,this._device=t,this._vertices=[],this._indices=[]}initialize(){const t=[];for(let n=0;n<this._vertices.length;n++){const{position:s,normal:a,tangent:o,texCoord:l}=this._vertices[n];t.push(...s,...a,...o,...l)}const r=new Float32Array(t);this._vertexBuffer=this._device.createBuffer({label:`${this._label}-vertex-buffer`,size:r.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST}),this._device.queue.writeBuffer(this._vertexBuffer,0,r);const i=new Uint32Array(this._indices);this._indexBuffer=this._device.createBuffer({label:`${this._label}-index-buffer`,size:i.byteLength,usage:GPUBufferUsage.INDEX|GPUBufferUsage.COPY_DST}),this._device.queue.writeBuffer(this._indexBuffer,0,i)}draw(t){if(!this._vertexBuffer||!this._indexBuffer){console.error(`${this._label} mesh was not initialized!`);return}if(!t){console.error(`GPURenderPassEncoder was not passed to ${this._label} mesh`);return}t.setVertexBuffer(0,this._vertexBuffer),t.setIndexBuffer(this._indexBuffer,"uint32"),t.drawIndexed(this._indices.length)}}class Le extends oe{constructor(t,r){super(t,"sphere");const i=500,n=500,s=-2*Math.PI/i,a=-1*Math.PI/n;for(let o=0;o<=n;o++){const l=V();Z(l,V(),a*o,h(0,0,1));const f=C();D(f,h(0,-r,0),l);for(let c=0;c<=i;c++){const u=V();Z(u,V(),s*c,h(0,1,0));const d=C();D(d,f,u);const p=C();se(p,d);const v=L(1-c/i,1-o/n);this._vertices.push({position:d,normal:p,texCoord:v,tangent:C()})}}for(let o=0;o<n;o++){const l=(i+1)*o;for(let f=0;f<i;f++){const c=l+f,u=l+f+1,d=l+(f+1)%(i+1)+(i+1),p=d-1;this._indices.push(c),this._indices.push(d),this._indices.push(u),this._indices.push(c),this._indices.push(p),this._indices.push(d);const v=this._vertices[c].position,m=this._vertices[u].position,g=this._vertices[d].position,x=this._vertices[p].position,y=this.calculateTangent(v,m,g,this._vertices[c].texCoord,this._vertices[u].texCoord,this._vertices[d].texCoord),w=this.calculateTangent(v,g,x,this._vertices[c].texCoord,this._vertices[d].texCoord,this._vertices[p].texCoord);this._vertices[c].tangent=y,this._vertices[u].tangent=y,this._vertices[d].tangent=y,this._vertices[c].tangent=w,this._vertices[d].tangent=w,this._vertices[p].tangent=w}}this.initialize()}calculateTangent(t,r,i,n,s,a){const o=C();J(o,r,t);const l=C();J(l,i,t);const f=X();ee(f,s,n);const c=X();ee(c,a,n);const u=1/(f[0]*c[1]-c[0]*f[1]),d=C();return d[0]=u*(c[1]*o[0]-f[1]*l[0]),d[1]=u*(c[1]*o[1]-f[1]*l[1]),d[2]=u*(c[1]*o[2]-f[1]*l[2]),d}}class Ve{constructor({position:t,center:r,up:i}){_(this,"_position");_(this,"_center");_(this,"_up");_(this,"_rotate");_(this,"_isMobile");_(this,"_isDragging");_(this,"_initialX");_(this,"_initialY");this._position=t,this._center=r,this._up=i,this._rotate=L(0,0),this._isDragging=!1,this._initialX=0,this._initialY=0,this._isMobile=/Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent),this.initializeEvent()}get position(){const t=this.getViewRotationMatrix(),r=C();return D(r,this._position,t),r}getViewMatrix(){const t=V(),r=this.getViewRotationMatrix(),i=C(),n=C(),s=C();return D(i,this._position,r),D(n,this._center,r),D(s,this._up,r),Ee(t,i,n,s),t}initializeEvent(){const t=this._isMobile?"touchstart":"mousedown",r=this._isMobile?"touchmove":"mousemove",i=this._isMobile?"touchend":"mouseup";document.addEventListener(t,n=>{this._isDragging=!0,this._initialX=this._isMobile?n.touches[0].clientX:n.clientX,this._initialY=this._isMobile?n.touches[0].clientY:n.clientY}),document.addEventListener(r,n=>{if(this._isDragging){const s=this._isMobile?n.touches[0].clientX:n.clientX,a=this._isMobile?n.touches[0].clientY:n.clientY,o=s-this._initialX,l=a-this._initialY;this._rotate=Ce(this._rotate,this._rotate,L(l/10,o/10)),this._initialX=s,this._initialY=a,n.preventDefault()}}),document.addEventListener(i,()=>{this._isDragging=!1})}getViewRotationMatrix(){const t=V();return re(t,t,j(this._rotate[1])),ie(t,t,j(this._rotate[0])),t}}class Ae{static getModelMatrix({translation:t,scaling:r,rotation:i}){const n=V();return ye(n,n,t),we(n,n,r),ie(n,n,i[0]),Me(n,n,i[1]),re(n,n,i[2]),n}}class I{static async loadImageBitmap(t){const i=await(await fetch(t)).blob();return await createImageBitmap(i,{colorSpaceConversion:"none"})}static async generateMips(t,r){let i=t;const n=[i];let s=0;for(;s<r&&(i.width>1||i.height>1);)i=await this.createNextMipLevelRgba8Unorm(i),n.push(i),s++;return n}static async createNextMipLevelRgba8Unorm(t){const r=Math.max(1,t.width/2|0),i=Math.max(1,t.height/2|0),n=document.createElement("canvas");n.width=r,n.height=i;const s=n.getContext("2d");if(!s)throw new Error("Unable to get 2D context");return s.drawImage(t,0,0,r,i),createImageBitmap(n)}}class A{constructor(t){_(this,"_device");_(this,"_texture");this._device=t}get view(){return this._texture||console.error("You need to initialize texture first!"),this._texture.createView()}async initialize(t,r=0){const i=await I.loadImageBitmap(t),n=await I.generateMips(i,r);if(this._texture=this._device.createTexture({label:"yellow F on red",size:[n[0].width,n[0].height],mipLevelCount:n.length,format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),!this._texture){console.error("Failed to load texture");return}n.forEach((s,a)=>{this._device.queue.copyExternalImageToTexture({source:s,flipY:!1},{texture:this._texture,mipLevel:a},{width:s.width,height:s.height})})}}class te{constructor(t){_(this,"_device");_(this,"_texture");this._device=t}get view(){return this._texture||console.error("You need to initialize texture first!"),this._texture.createView({dimension:"cube"})}async initialize(t,r=0){const i=await Promise.all(t.map(I.loadImageBitmap));if(this._texture=this._device.createTexture({label:"yellow F on red",size:[i[0].width,i[0].height,i.length],mipLevelCount:r+1,format:"rgba8unorm",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT}),!this._texture){console.error("Failed to load texture");return}for(let n=0;n<6;n++)(await I.generateMips(i[n],r)).forEach((a,o)=>{this._device.queue.copyExternalImageToTexture({source:a,flipY:!1},{texture:this._texture,origin:[0,0,n],mipLevel:o},{width:a.width,height:a.height})})}}class De extends oe{constructor(t){super(t,"skybox"),this._vertices.push({position:h(-1,1,-1),normal:h(0,0,0),tangent:h(0,0,0),texCoord:L(0,0)}),this._vertices.push({position:h(-1,-1,-1),normal:h(0,0,0),tangent:h(0,0,0),texCoord:L(0,0)}),this._vertices.push({position:h(1,1,-1),normal:h(0,0,0),tangent:h(0,0,0),texCoord:L(0,0)}),this._vertices.push({position:h(1,-1,-1),normal:h(0,0,0),tangent:h(0,0,0),texCoord:L(0,0)}),this._vertices.push({position:h(-1,1,1),normal:h(0,0,0),tangent:h(0,0,0),texCoord:L(0,0)}),this._vertices.push({position:h(1,1,1),normal:h(0,0,0),tangent:h(0,0,0),texCoord:L(0,0)}),this._vertices.push({position:h(-1,-1,1),normal:h(0,0,0),tangent:h(0,0,0),texCoord:L(0,0)}),this._vertices.push({position:h(1,-1,1),normal:h(0,0,0),tangent:h(0,0,0),texCoord:L(0,0)}),this._indices.push(0,1,2,2,1,3,2,3,5,5,3,7,5,7,4,4,7,6,4,6,0,0,6,1,4,0,5,5,0,2,1,6,3,3,6,7),this.initialize()}}const W=document.documentElement.clientWidth,ne=document.documentElement.clientHeight;async function Oe(){const e=new he(document.querySelector("canvas"));if(await e.initialize(W,ne),!e.device)return;const t="",r=new A(e.device);await r.initialize(t+"pbr/antique-grate1-height.jpg");const i=new A(e.device);await i.initialize(t+"pbr/antique-grate1-albedo.jpg");const n=new A(e.device);await n.initialize(t+"pbr/antique-grate1-normal-dx.jpg");const s=new A(e.device);await s.initialize(t+"pbr/antique-grate1-metallic.jpg");const a=new A(e.device);await a.initialize(t+"pbr/antique-grate1-roughness.jpg");const o=new A(e.device);await o.initialize(t+"pbr/antique-grate1-ao.jpg");const l=new A(e.device);await l.initialize(t+"cubemap/air_museum_playground_brdf.jpg");const f=new te(e.device);await f.initialize([t+"cubemap/air_museum_playground_env_px.jpg",t+"cubemap/air_museum_playground_env_nx.jpg",t+"cubemap/air_museum_playground_env_py.jpg",t+"cubemap/air_museum_playground_env_ny.jpg",t+"cubemap/air_museum_playground_env_pz.jpg",t+"cubemap/air_museum_playground_env_nz.jpg"]);const c=new te(e.device);await c.initialize([t+"cubemap/air_museum_playground_irradiance_px.jpg",t+"cubemap/air_museum_playground_irradiance_nx.jpg",t+"cubemap/air_museum_playground_irradiance_py.jpg",t+"cubemap/air_museum_playground_irradiance_ny.jpg",t+"cubemap/air_museum_playground_irradiance_pz.jpg",t+"cubemap/air_museum_playground_irradiance_nz.jpg"],5);const u=e.device.createSampler({magFilter:"linear",minFilter:"linear",mipmapFilter:"linear"}),d=new Le(e.device,W>=500?.6:.4),p=new De(e.device),v=new K({label:"main",device:e.device,vertexShader:de,fragmentShader:fe,bindGroupLayouts:[e.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},{binding:2,visibility:GPUShaderStage.FRAGMENT|GPUShaderStage.VERTEX,sampler:{type:"filtering"}},{binding:3,visibility:GPUShaderStage.VERTEX,texture:{sampleType:"float",viewDimension:"2d"}},{binding:4,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float",viewDimension:"2d"}},{binding:5,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float",viewDimension:"2d"}},{binding:6,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float",viewDimension:"2d"}},{binding:7,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float",viewDimension:"2d"}},{binding:8,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float",viewDimension:"2d"}},{binding:9,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float",viewDimension:"2d"}},{binding:10,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float",viewDimension:"cube"}},{binding:11,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float",viewDimension:"cube"}}]})]}),m=new K({label:"bg",device:e.device,vertexShader:ue,fragmentShader:pe,bindGroupLayouts:[e.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX,buffer:{type:"uniform"}},{binding:1,visibility:GPUShaderStage.FRAGMENT,sampler:{type:"filtering"}},{binding:2,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:"float",viewDimension:"cube"}}]})]}),g=e.device.createBuffer({label:"vertex uniforms",size:64*Float32Array.BYTES_PER_ELEMENT,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),x=e.device.createBuffer({label:"fragment uniforms",size:4*Float32Array.BYTES_PER_ELEMENT,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),y=e.device.createBuffer({label:"vertex uniforms",size:32*Float32Array.BYTES_PER_ELEMENT,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),w=e.device.createBindGroup({label:"bind group for object",layout:v.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:g}},{binding:1,resource:{buffer:x}},{binding:2,resource:u},{binding:3,resource:r.view},{binding:4,resource:i.view},{binding:5,resource:n.view},{binding:6,resource:s.view},{binding:7,resource:a.view},{binding:8,resource:o.view},{binding:9,resource:l.view},{binding:10,resource:f.view},{binding:11,resource:c.view}]}),E=e.device.createBindGroup({label:"bind group for object",layout:m.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:y}},{binding:1,resource:u},{binding:2,resource:f.view}]}),P=h(0,0,0),G=h(1,1,1),R=h(0,0,0),N=new Ve({position:h(0,0,2.5),center:h(0,0,0),up:h(0,1,0)}),U=V();Te(U,j(45),W/ne,.1,100);function z(){var k,H,$;const F=Ae.getModelMatrix({translation:P,scaling:G,rotation:R}),B=N.getViewMatrix(),M=me(F);xe(M,M),be(M,M),(k=e.device)==null||k.queue.writeBuffer(g,0,new Float32Array([...F,...B,...U,...M])),(H=e.device)==null||H.queue.writeBuffer(y,0,new Float32Array([...B,...U]));const b=Pe(N.position);($=e.device)==null||$.queue.writeBuffer(x,0,new Float32Array([...b,0]));const S=e.getPass();m.use(S),S==null||S.setBindGroup(0,E),p.draw(S),v.use(S),S==null||S.setBindGroup(0,w),d.draw(S),e.endPass(),requestAnimationFrame(z)}z()}Oe();
