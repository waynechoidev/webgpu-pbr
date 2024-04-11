#include "main.common.wgsl"
#include "main.frag.utils.wgsl"

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
  // Apply gamma correction to the sampled albedo texture to convert it from sRGB space to linear space
  let albedo:vec3f = pow(textureSample(albedoMap, sampler2D, input.texCoord).rgb, vec3(2.2));
  let metallic:f32 = textureSample(metallicMap, sampler2D, input.texCoord).r;
  let roughness:f32 = textureSample(roughnessMap, sampler2D, input.texCoord).r;
  let ao:f32 = textureSample(aoMap, sampler2D, input.texCoord).r;

  let normalWorld:vec3f = normalize(input.normalWorld);
  // Adjust the tangent vector to ensure it is perpendicular to the surface
  // by removing the component parallel to the normal vector.
  let tangent:vec3f = normalize(input.tangentWorld - dot(input.tangentWorld, normalWorld) * normalWorld);
  let bitangent:vec3f = cross(normalWorld, tangent);
  let TBN:mat3x3f = mat3x3f(tangent, bitangent, normalWorld);
  let N:vec3f = normalize(TBN * (textureSample(normalMap, sampler2D, input.texCoord).xyz * 2.0 - 1.0));

  let V:vec3f = normalize(uni.camPos - input.posWorld);
  let R:vec3f = reflect(-V, N);

  let lightRadiances:vec3f = vec3f(10.0);

  var F0:vec3f = vec3f(0.04);
  F0 = mix(F0, albedo, metallic);

  // use camPos instead of lightPos
  let L:vec3f = normalize(uni.camPos - input.posWorld);
  let H:vec3f = normalize(V + L);
  let distance:f32 = length(uni.camPos - input.posWorld);
  let attenuation:f32 = 1.0 / (distance * distance);
  let radiance:vec3f = lightRadiances * attenuation;

  let F:vec3f = fresnelSchlick(max(dot(H, V), 0.0), F0);
  let G:f32 = GeometrySmith(N, V, L, roughness);
  let NDF:f32 = DistributionGGX(N, H, roughness);

  let numerator:vec3f = NDF * G * F;
  // + 0.0001 to prevent divide by zero
  let denominator:f32 = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
  let specular:vec3f = numerator / denominator;

  let kS:vec3f = F;

  let kD:vec3f = (vec3(1.0) - kS) * (1.0 - metallic);

  // scale light by NdotL
  let NdotL:f32 = max(dot(N, L), 0.0);

  let directLight:vec3f = (kD * albedo / PI + specular) * radiance * NdotL;

  // // ambient lighting (we now use IBL as the ambient term)
  // F = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);

  // kS = F;
  // kD = 1.0 - kS;
  // kD *= 1.0 - metallic;

  // // Apply gamma correction to the sampled irradianceCubemap texture to convert it from sRGB space to linear space
  // vec3 irradiance = pow(texture(irradianceCubemap, N).rgb, vec3(2.2));
  // vec3 diffuse = irradiance * albedo;

  // // sample both the pre-filter map and the BRDF lut and combine them together as per the Split-Sum approximation to get the IBL specular part.
  // const float MAX_REFLECTION_LOD = 4.0;
  // // Apply gamma correction to the sampled envCubemap texture to convert it from sRGB space to linear space
  // vec3 prefilteredColor = pow(textureLod(envCubemap, R,  roughness * MAX_REFLECTION_LOD).rgb, vec3(2.2));
  // vec2 brdf  = texture(brdfLUT, vec2(max(dot(N, V), 0.0), roughness)).rg;
  // specular = prefilteredColor * (F * brdf.x + brdf.y);

  // vec3 ambient = (kD * diffuse + specular) * ao;

  var color:vec3f = directLight;// + ambient;

  // HDR tonemapping
  color = color / (color + vec3(1.0));
  // gamma correct
  color = pow(color, vec3(1.0/2.2));

  return vec4f(color, 1.0);

  // return textureSampleLevel(albedoMap, sampler2D, input.texCoord, 0);
  // return vec4f(albedo, 1.0);
}