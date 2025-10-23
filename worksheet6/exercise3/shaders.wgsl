// Structures

// For image coordinates
struct VSOut {
    @builtin(position) position: vec4f,
    @location(0) coords : vec2f,
};
// For JS variables
struct Uniforms {
    aspect: f32,
    cam_const: f32,
    gamma: f32,
    _pad0: f32,
    eye: vec3f, _pad1: f32,
    v: vec3f, _pad2: f32,
    b1: vec3f, _pad3: f32,
    b2: vec3f, _pad4: f32,
    matt_shader: f32,
    repeat_selector: f32,
    filter_selector: f32,
    _pad5: f32,
    texture_enable: f32,
    subdivs: f32,
    texture_scale: f32,
    _pad7: f32,
};
// For storing coordinate info
struct HitInfo { 
    has_hit: bool, 
    dist: f32, 
    position: vec3f, 
    normal: vec3f, 
    color: vec3f,
    diffuse: vec3f,
    emission: vec3f,
    specular: vec3f, 
    shader: i32, 
    continue_path: bool,
    iof: f32
}; 
// For ray tracing
struct Ray {
    origin: vec3f,
    direction: vec3f,
    tmin: f32,
    tmax: f32
}
// For color mapping
struct Light { 
    L_i: vec3f, 
    w_i: vec3f, 
    dist: f32 
}; 
// For the plane (Orthonormal basis)
struct Onb { 
    tangent: vec3f, 
    binormal: vec3f, 
    normal: vec3f, 
}; 
struct Material {
    emission: vec4f,
    diffuse: vec4f,
};
struct Aabb {
    min: vec3f,
    max: vec3f,
};
struct Attribs {
    position: vec4f,
    normal: vec4f,
};

// Bindings from JS

// Basic info
@group(0) @binding(0) var<uniform> uniforms : Uniforms;
// Anti-alias
@group(0) @binding(1) var<storage> jitter: array<vec2f>;
// Mesh data
@group(0) @binding(2) var<storage> attributes: array<Attribs>;
@group(0) @binding(3) var<storage> meshFaces: array<vec4u>; // v0,v1,v2,mat
// Material indices: one u32 per triangle
@group(0) @binding(4) var<uniform> aabb : Aabb;
// BSP tree buffers (produced by build_bsp_tree)
@group(0) @binding(5) var<storage> materials: array<Material>;
@group(0) @binding(6) var<storage> bspTree: array<u32>; // node*4 layout
@group(0) @binding(7) var<storage> bspPlanes: array<f32>;
@group(0) @binding(8) var<storage> treeIds: array<u32>;
@group(0) @binding(9) var<storage> lightIndices: array<u32>;


const MAX_LEVEL = 20u;
const BSP_LEAF = 3u;
var<private> branch_node: array<vec2u, MAX_LEVEL>;
var<private> branch_ray: array<vec2f, MAX_LEVEL>;

// AABB intersection
fn intersect_min_max(r: ptr<function, Ray>) -> bool {
    let p1 = (aabb.min - r.origin)/r.direction;
    let p2 = (aabb.max - r.origin)/r.direction;
    let pmin = min(p1, p2);
    let pmax = max(p1, p2);
    let box_tmin = max(pmin.x, max(pmin.y, pmin.z)) - 1.0e-3f;
    let box_tmax = min(pmax.x, min(pmax.y, pmax.z)) + 1.0e-3f;
    if(box_tmin > box_tmax || box_tmin > r.tmax || box_tmax < r.tmin) {
        return false;
    }
    r.tmin = max(box_tmin, r.tmin);
    r.tmax = min(box_tmax, r.tmax);
    return true;
}
// Intersection functions
fn intersect_plane(r: Ray, hit: ptr<function, HitInfo>, position: vec3f, onb: Onb) -> bool {
    let t = -dot(r.origin, onb.normal) / dot(r.direction, onb.normal);
    let p = r.origin + r.direction * t;
    let has_hit = r.tmin < t && t < r.tmax;
    if has_hit {
        (*hit).position = p;
        (*hit).dist = t; 
        (*hit).normal = normalize(onb.normal);
        (*hit).has_hit = true;

        (*hit).shader = i32(uniforms.matt_shader);

        return true;
    } else {
        return false;
    };
}
fn intersect_triangle(r: Ray, hit: ptr<function, HitInfo>, triIndex: u32) -> bool {
    // Read triangle vertex indices (meshFaces stores vec4u: x,y,z,mat)
    let face = meshFaces[triIndex];
    let p0 = attributes[face.x].position.xyz;
    let p1 = attributes[face.y].position.xyz;
    let p2 = attributes[face.z].position.xyz;

    let e0 = p1 - p0;
    let e1 = p2 - p0;
    let n = cross(e0, e1);

    let denom = dot(r.direction, n);
    let eps = 1e-8;
    if (abs(denom) < eps) { return false; }

    // Distance along ray to plane of triangle
    let t = dot(p0 - r.origin, n) / denom;
    if (!(r.tmin < t && t < r.tmax)) { return false; }

    // Barycentric coordinates
    let a = p0 - r.origin;
    let c1 = cross(a, r.direction);
    let beta = dot(c1, e1) / denom;
    let gamma = -dot(c1, e0) / denom;

    if (beta >= 0.0 && gamma >= 0.0 && (beta + gamma) <= 1.0) {
        // Interpolate vertex normals (if present) for smooth shading
        let w0 = 1.0 - beta - gamma;
        let n0 = attributes[face.x].normal.xyz;
        let n1 = attributes[face.y].normal.xyz;
        let n2 = attributes[face.z].normal.xyz;
        let interpN = normalize(n0 * w0 + n1 * beta + n2 * gamma);

        (*hit).normal = interpN;

        // Material lookup (guard against out-of-bounds)
        let mid = meshFaces[triIndex].w;
        if (mid < arrayLength(&materials)) {
            let mat = materials[mid];
            (*hit).diffuse = mat.diffuse.xyz;
            (*hit).emission = mat.emission.xyz;
            (*hit).color = mat.diffuse.xyz + mat.emission.xyz;
        }

        (*hit).has_hit = true;
        (*hit).dist = t;
        (*hit).position = r.origin + r.direction * t;
        (*hit).shader = i32(uniforms.matt_shader);
        return true;
    }
    return false;
}

fn intersect_sphere(r: Ray, hit: ptr<function, HitInfo>, center: vec3f, radius: f32) -> bool {
    let oc = r.origin - center;
    let b = dot(oc, r.direction);
    let c = dot(oc, oc) - radius * radius;
    let discriminant = b * b - c;
    if discriminant < 0.0 {

        return false;
    }
    let sqrt_disc = sqrt(discriminant);
    let t1 = -b - sqrt_disc;
    let t2 = -b + sqrt_disc;
    var t = t1;
    if t < r.tmin || t > r.tmax {
        t = t2;
        if t < r.tmin || t > r.tmax {
            return false;
        }
    } 
    (*hit).dist = t;
    (*hit).position = r.origin + r.direction * t;
    (*hit).normal = normalize((*hit).position - center);
    (*hit).has_hit = true;

    (*hit).shader = 0;

    return true;
}
fn intersect_trimesh(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> bool{
    var branch_lvl = 0u;
    var near_node = 0u;
    var far_node = 0u;
    var t = 0.0f;
    var node = 0u;
    for(var i = 0u; i <= MAX_LEVEL; i++) {
        let nodeBase = node * 4u;
        let nodeInfo = bspTree[nodeBase];
        let node_axis_leaf = nodeInfo & 3u;
        if(node_axis_leaf == BSP_LEAF) {
            // A leaf was found
            let node_count = nodeInfo >> 2u;
            let node_id = bspTree[nodeBase + 1u];
            var found = false;
            for (var j = 0u; j < node_count; j++) {
                let obj_idx = treeIds[node_id + j];
                if (intersect_triangle(*r, hit, obj_idx)) {
                    r.tmax = hit.dist;
                    found = true;
                }
            }
            if (found) { return true; }
            else if (branch_lvl == 0u) { return false; }
            else {
                branch_lvl = branch_lvl - 1u;
                i = branch_node[branch_lvl].x;
                node = branch_node[branch_lvl].y;
                r.tmin = branch_ray[branch_lvl].x;
                r.tmax = branch_ray[branch_lvl].y;
                continue;
            }
        }
        let left = bspTree[nodeBase + 2u];
        let right = bspTree[nodeBase + 3u];
        let axis_direction = r.direction[node_axis_leaf];
        let axis_origin = r.origin[node_axis_leaf];
        if(axis_direction >= 0.0f) {
            near_node = left;
            far_node = right;
        } else {
            near_node = right;
            far_node = left;
        }
        let node_plane = bspPlanes[node];
        let denom = select(axis_direction, 1.0e-8f, abs(axis_direction) < 1.0e-8f);
        t = (node_plane - axis_origin)/denom;
        if(t > r.tmax) { node = near_node; }
        else if(t < r.tmin) { node = far_node; }
        else {
            branch_node[branch_lvl].x = i;
            branch_node[branch_lvl].y = far_node;
            branch_ray[branch_lvl].x = t;
            branch_ray[branch_lvl].y = r.tmax;
            branch_lvl = branch_lvl + 1u;
            r.tmax = t;
            node = near_node;
        }
    }
    return false;
}
// Top-level scene intersection: AABB culling + trimesh traversal
fn intersect_scene(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> bool {
    if (!intersect_min_max(r)) { return false; }
    return intersect_trimesh(r, hit);
}
// Light sample for color calculations
fn sample_point_light(pos: vec3f) -> Light {
    var pi = 3.14159;
    let light_origin = vec3f(0.0, 1.0, 0.0);
    let light = Light(
        vec3f(pi, pi, pi),
        normalize(light_origin - pos),
        length(light_origin - pos)
    );
    return light;
}
// Sum radiometric contributions from emissive triangles (lightIndices).
fn sample_area_lights(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    var sum = vec3f(0.0);
    let nLights = arrayLength(&lightIndices);
    if (nLights == 0u) { return sum; }

    let Nsurf = normalize((*hit).normal);
    // Loop over emissive triangles
    for (var li = 0u; li < nLights; li = li + 1u) {
        let triIndex = lightIndices[li];
        let idx = meshFaces[triIndex];
        let triangle_v = array<vec3f, 3>(
            attributes[idx.x].position.xyz,
            attributes[idx.y].position.xyz,
            attributes[idx.z].position.xyz
        );

        let e0 = triangle_v[1] - triangle_v[0];
        let e1 = triangle_v[2] - triangle_v[0];
        let normal = normalize(cross(e0, e1));
        let area = 0.5 * length(cross(e0, e1));

        let centroid = (triangle_v[0] + triangle_v[1] + triangle_v[2]) * (1.0 / 3.0);
        let toLight = centroid - (*hit).position;

        let dist = length(toLight);
        // skip degenerate/very-close lights
        if (dist <= 1e-6) { continue; }
        let wi = normalize(toLight);

        // Cosine on light side: how much the patch faces the point
        let cosL = max(dot(normal, -wi), 0.0);
        if (cosL <= 0.0) { continue; }

        // Cosine on surface side
        let cosS = max(dot(Nsurf, wi), 0.0);
        if (cosS <= 0.0) { continue; }

        var shadow_tmax = max(dist - 1.0, 1.0);
        var shadow = Ray((*hit).position, wi, 1.0, shadow_tmax);
        var shHit: HitInfo;
        if (intersect_scene(&shadow, &shHit)) { continue; }

        let mid = meshFaces[triIndex].w;
        var Le = vec3f(0.0);
        if (mid < arrayLength(&materials)) {
            Le = materials[mid].emission.xyz;
        }

        // Radiometric approx: irradiance contribution ~ Le * (area * cosL) / dist^2
        // Reflected radiance for Lambertian: (Rd / pi) * E * cosS
        let E = Le * (area * cosL) / (dist * dist);
        let brdf = (*hit).diffuse / 3.14159;
        let contrib = brdf * E * cosS;
        sum = sum + contrib;
    }
    return sum;
}
// Shade functions
fn area_lights(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    // sample_area_lights already returns reflected radiance (BRDF * E * cos), so return it directly
    return sample_area_lights(r, hit);
}
fn directional_light(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    let pi = 3.14159;
    let intensity = vec3f(pi);
    let dir = -normalize(vec3f(-1.0, -1.0, -1.0));

    let N = normalize((*hit).normal);
    let nwi = max(dot(N, dir), 0.0);

    let Li = intensity;
    let Lr = (
        hit.emission + 
        hit.diffuse / pi
    ) * Li * nwi;

    return Lr;
}
fn lambertian(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    let light = sample_point_light((*hit).position);

    var shadow = Ray(hit.position, light.w_i, 1e-4, max(light.dist - 1e-4, 0.0));
    var shadow_hit: HitInfo;

    if intersect_scene(&shadow, &shadow_hit) {
        return hit.emission;
    };

    let N = normalize((*hit).normal);
    let nwi = max(dot(N, light.w_i), 0.0);
    let Li = light.L_i / (light.dist * light.dist);

    let pi = 3.14159;
    let Lr = (
        hit.emission + 
        hit.diffuse / pi
    ) * Li * nwi;
    return Lr;
}
fn mirror(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {

    (*hit).continue_path = true;
    (*hit).has_hit = false;
    (*r).origin = hit.position;
    (*r).direction = reflect(r.direction, hit.normal);
    (*r).tmin = 1e-4;
    (*r).tmax = 100.0;

    return vec3f(0.0);
}
fn refract_shader(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    let n_air = 1.0;
    let n_glass = 1.5;
    let N = normalize((*hit).normal);
    let V = normalize((*r).direction);

    // Determine if entering or exiting
    var normal = N;
    var n1 = n_air;
    var n2 = n_glass;

    let dir_vector = dot(V, N);
    if (dir_vector > 0.0) {
        // swap indices and flip normal
        normal = -N;
        n1 = n_glass;
        n2 = n_air;
    }
    let eta = n1 / n2;
    let refracted = refract(V, normal, eta);

    (*hit).continue_path = true;
    (*r).origin = hit.position;
    (*r).direction = refracted;
    (*r).tmin = 1e-4;
    (*r).tmax = 100.0;
    (*hit).iof = n2;

    return vec3f(0.0);
}
fn phong(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    let light = sample_point_light((*hit).position);
    let N = normalize((*hit).normal);
    let V = normalize(-(*r).direction);
    let L = normalize(light.w_i);
    let R = reflect(-L, N); 
    let nwi = max(dot(N, L), 0.0);
    let Li = light.L_i / (light.dist * light.dist);
    let s = 42.0; 
    let pi = 3.14159;

    let diffuse = hit.diffuse / pi;
    let spec_angle = max(dot(R, V), 0.0);
    let specular = hit.specular * pow(spec_angle, s);

    let Lr = (hit.emission + diffuse + specular) * Li * nwi;
    return Lr;
}
fn glossy(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    let phong_shading = phong(r, hit);
    let refract_shading = refract_shader(r, hit);
    return phong_shading + refract_shading;
}
fn shade(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f { 
    switch hit.shader { 
        case 0: { return hit.color; }
        case 1: { return directional_light(r, hit); }
        case 2: { return lambertian(r, hit); } 
        case 3: { return area_lights(r, hit); } 
        case 4: { return mirror(r, hit); } 
        case 5: { return refract_shader(r, hit); }
        case 6: { return phong(r, hit); }
        case 7: { return glossy(r, hit); }
        case default { return vec3f(0.0); } 
    } 
    
}

// Texture look-up
fn texture_nearest(texture: texture_2d<f32>, texcoords: vec2f, repeat: bool) -> vec3f {
    let WH = textureDimensions(texture);
    var st = texcoords;
    if (repeat) {
        st = texcoords - floor(texcoords); // Repeat mode
    } else {
        st = clamp(texcoords, vec2f(0.0, 0.0), vec2f(1.0, 1.0)); // Clamp-to-edge mode
    }
    let ab = st * vec2f(WH);
    let uv = vec2u(ab + 0.5) % WH;
    let texcolor = textureLoad(texture, uv, 0);
    return texcolor.rgb;
}
fn texture_linear(texture: texture_2d<f32>, texcoords: vec2f, repeat: bool) -> vec3f {
    let WH = textureDimensions(texture);
    var st = texcoords;
    if (repeat) {
        st = texcoords - floor(texcoords); // Repeat mode
    } else {
        st = clamp(texcoords, vec2f(0.0, 0.0), vec2f(1.0, 1.0)); // Clamp-to-edge mode
    }
    let ab = st * vec2f(WH) - 0.5;
    let i = floor(ab);
    let f = ab - i;
    var i0 = vec2i(i);
    var i1 = i0 + vec2i(1, 0);
    var i2 = i0 + vec2i(0, 1);
    var i3 = i0 + vec2i(1, 1);

    // Handle repeat or clamp for indices
    let W = i32(WH.x);
    let H = i32(WH.y);
    if (repeat) {
        i0 = vec2i((i0.x % W + W) % W, (i0.y % H + H) % H);
        i1 = vec2i((i1.x % W + W) % W, (i1.y % H + H) % H);
        i2 = vec2i((i2.x % W + W) % W, (i2.y % H + H) % H);
        i3 = vec2i((i3.x % W + W) % W, (i3.y % H + H) % H);
    } else {
        i0 = clamp(i0, vec2i(0), vec2i(W-1, H-1));
        i1 = clamp(i1, vec2i(0), vec2i(W-1, H-1));
        i2 = clamp(i2, vec2i(0), vec2i(W-1, H-1));
        i3 = clamp(i3, vec2i(0), vec2i(W-1, H-1));
    }

    let c00 = textureLoad(texture, vec2u(i0), 0).rgb;
    let c10 = textureLoad(texture, vec2u(i1), 0).rgb;
    let c01 = textureLoad(texture, vec2u(i2), 0).rgb;
    let c11 = textureLoad(texture, vec2u(i3), 0).rgb;

    let cx0 = mix(c00, c10, f.x);
    let cx1 = mix(c01, c11, f.x);
    let c = mix(cx0, cx1, f.y);
    return c;
}
fn texture_switch(texture: texture_2d<f32>, texcoords: vec2f) -> vec3f {
    switch i32(uniforms.filter_selector) {
        case 0: {return texture_nearest(texture, texcoords, uniforms.repeat_selector == 0); }
        case 1: {return texture_linear(texture, texcoords, uniforms.repeat_selector == 0); }
        case default: { return texture_nearest(texture, texcoords, uniforms.repeat_selector == 0); }
    }
}

// Initialize ray tracing
fn trace_rays(ipcoords: vec2f, uniforms: Uniforms) -> Ray {
    let xip = ipcoords.x * uniforms.aspect * 0.5f;
    let yip = ipcoords.y * 0.5f;

    let qb1 = uniforms.b1 * xip;
    let qb2 = uniforms.b2 * yip;
    let qv = uniforms.v * uniforms.cam_const;

    let origin = uniforms.eye;
    let direction = normalize((qb1 + qb2 + qv));

    return Ray(
        origin,
        direction,
        0.0,
        2000.0 // increased to cover larger scene distances (Cornell box)
    );
}

@vertex
fn main_vs(@builtin(vertex_index) VertexIndex : u32) -> VSOut {
    const pos = array<vec2f, 4>(vec2f(-1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(1.0, -1.0));
    var vsOut: VSOut;
    vsOut.position = vec4f(pos[VertexIndex], 0.0, 1.0);
    vsOut.coords = pos[VertexIndex];
    return vsOut;
}


@fragment
fn main_fs(@location(0) coords: vec2f) -> @location(0) vec4f {
    const bgcolor = vec4f(0.1, 0.3, 0.6, 1.0);
    const max_depth = 10;
    var result = vec3f(0.0);
    var n_samples = i32(uniforms.subdivs * uniforms.subdivs);
    if (n_samples < 1) { n_samples = 1; }
    for (var s = 0; s < n_samples; s = s + 1) {
        // Get jitter offset for this sample
        let jitter_offset = jitter[s]; // vec2f
        // Apply jitter to pixel coordinates
        let jittered_coords = coords + jitter_offset;
        var r = trace_rays(jittered_coords, uniforms);
        var sample_result = vec3f(0.0);
        var hit: HitInfo;
        for (var i = 0; i < max_depth; i = i + 1) {
            if (intersect_scene(&r, &hit)) { sample_result += shade(&r, &hit); }
            else { sample_result += bgcolor.rgb; break; }
            if (hit.has_hit && !hit.continue_path) { break; }
        }
        result += sample_result;
    }
    result = result / f32(n_samples);
    return vec4f(pow(result, vec3f(1.0/uniforms.gamma)), bgcolor.a);
}