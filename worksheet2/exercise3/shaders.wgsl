struct VSOut {
    @builtin(position) position: vec4f,
    @location(0) coords : vec2f,
};

struct Uniforms {
    aspect: f32,
    cam_const: f32,
    gamma: f32,
    _pad0: f32,
    eye: vec3f, _pad1: f32,
    v: vec3f, _pad2: f32,
    b1: vec3f, _pad3: f32,
    b2: vec3f, _pad4: f32,
    glass_shader: f32,
    matt_shader: f32,
    _pad5: f32,
    _pad6: f32,
};
@group(0) @binding(0) var<uniform> uniforms : Uniforms;

struct HitInfo { 
    has_hit: bool, 
    dist: f32, 
    position: vec3f, 
    normal: vec3f, 
    color: vec3f,
    diffuse: vec3f,
    ambient: vec3f,
    specular: vec3f, 
    shader: i32, 
    continue_path: bool,
    ior: f32,
}; 

// Define Ray struct
struct Ray {
    origin: vec3f,
    direction: vec3f,
    tmin: f32,
    tmax: f32
}

struct Light { 
    L_i: vec3f, 
    w_i: vec3f, 
    dist: f32 
}; 

fn intersect_plane(r: Ray, hit: ptr<function, HitInfo>, position: vec3f, normal: vec3f) -> bool {
    let t = -dot(r.origin, normal) / dot(r.direction, normal);
    let p = r.origin + r.direction * t;
    let has_hit = r.tmin < t && t < r.tmax;
    if has_hit {
        (*hit).color = vec3f(0.1, 0.7, 0.0);
        (*hit).ambient = 0.1 * (*hit).color ;
        (*hit).diffuse = 0.9 * (*hit).color ;
        (*hit).specular = vec3f(0.0, 0.0, 0.0);

        (*hit).position = p;
        (*hit).dist = t; 
        (*hit).normal = normalize(normal);
        (*hit).has_hit = true;
        (*hit).specular = vec3f(0.0, 0.0, 0.0);

        (*hit).shader = i32(uniforms.matt_shader);

        return true;
    } else {
        return false;
    };
}
fn intersect_triangle(r: Ray, hit: ptr<function, HitInfo>, v: array<vec3f, 3>) -> bool {
    let e0 = v[1] - v[0];
    let e1 = v[2] - v[0];
    let n = cross(e0, e1);
    
    let denom = dot(r.direction, n);
    if (denom == 0.0) { return false; }

    let t = dot((v[0] - r.origin), n) / denom;
    if !(r.tmin < t && t < r.tmax) { return false; }

    let a = v[0] - r.origin; 
    let beta  = dot(cross(a, r.direction), e1) / denom;
    let gamma = -dot(cross(a, r.direction), e0) / denom;

    if beta >= 0 && gamma >= 0 && (beta + gamma) <= 1 {
        (*hit).color  = vec3f(0.4,0.3,0.2);
        (*hit).ambient = 0.1 * (*hit).color ;
        (*hit).diffuse = 0.9 * (*hit).color ;
        (*hit).specular = vec3f(0.0, 0.0, 0.0);

        (*hit).normal = n;
        (*hit).has_hit = true;
        (*hit).dist = t;
        (*hit).position = r.origin + r.direction * (*hit).dist;

        (*hit).shader = i32(uniforms.matt_shader);

        return true;
    } else {
        return false;
    };
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
    (*hit).color  = vec3f(0.0, 0.0, 0.0);
    (*hit).ambient = 0.1 * (*hit).color ;
    (*hit).diffuse = 0.9 * (*hit).color ;
    (*hit).specular = vec3f(0.1, 0.1, 0.1);

    (*hit).dist = t;
    (*hit).position = r.origin + r.direction * t;
    (*hit).normal = normalize((*hit).position - center);
    (*hit).has_hit = true;

    (*hit).shader = i32(uniforms.glass_shader);

    return true;
}

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

fn lambertian(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    let light = sample_point_light((*hit).position);

    var shadow = Ray(hit.position, light.w_i, 1e-4, max(light.dist - 1e-4, 0.0));
    var shadow_hit: HitInfo;

    if intersect_scene(&shadow, &shadow_hit) {
        return hit.ambient;
    };

    let N = normalize((*hit).normal);
    let nwi = max(dot(N, light.w_i), 0.0);
    let Li = light.L_i / (light.dist * light.dist);

    let pi = 3.14159;
    let Lr = (
        hit.ambient + 
        hit.diffuse / pi
    ) * Li * nwi;
    return Lr;
}

fn mirror(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    // Set the flag to continue the path
    (*hit).continue_path = true;
    (*r).origin = hit.position;
    (*r).direction = reflect(r.direction, hit.normal);
    (*r).tmin = 1e-4;
    (*r).tmax = 100.0;
    // No direct color contribution for perfect mirror
    return vec3f(0.0);
}

fn refract_shader(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f {
    let n_air = 1.0;
    let n_glass = 1.5;
    let N = normalize((*hit).normal);
    let V = normalize((*r).direction);

    // Determine if entering or exiting
    let cos_theta = dot(N, V);
    var n1 = n_air;
    var n2 = n_glass;
    var normal = N;
    if (cos_theta > 0.0) {
        // Exiting: swap indices and flip normal
        normal = -N;
        n1 = n_glass;
        n2 = n_air;
    }

    let eta = n1 / n2;
    let refracted = refract(V, normal, eta);

    // Set up for next path
    (*hit).continue_path = true;
    (*r).origin = (*hit).position;
    (*r).direction = refracted;
    (*r).tmin = 1e-4;
    (*r).tmax = 100.0;
    (*hit).ior = n2;

    // No direct color contribution for perfect glass
    return vec3f(0.0);
}

fn shade(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> vec3f { 
    switch hit.shader { 
        case 0: { return hit.color; }
        case 1: { return lambertian(r, hit); } 
        case 2: { return mirror(r, hit); } 
        case 3: { return refract_shader(r, hit); }
        case default { return vec3f(0.0); } 
    } 
    
}

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
        100.0 // arbitrary
    );
}

fn intersect_scene(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> bool {
    var has_hit = false;
    (*hit).has_hit = false;

    // plane
    let plane_position = vec3f(0.0, 0.0, 0.0);
    let plane_normal = vec3f(0.0, 1.0, 0.0);
    intersect_plane(*r, hit, plane_position, plane_normal);

    // triangle
    let triangle_v = array<vec3f, 3>(
        vec3f(-0.2,0.1,0.9), 
        vec3f(0.2,0.1,0.9), 
        vec3f(-0.2,0.1,-0.1)
    );
    intersect_triangle(*r, hit, triangle_v);

    // sphere
    let sphere_center = vec3f(0.0, 0.5, 0.0);
    let sphere_radius = 0.3;
    intersect_sphere(*r, hit, sphere_center, sphere_radius);

    return hit.has_hit; 
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
fn main_fs(@location(0) coords: vec2f) -> @location(0) vec4f 
{ 
    const bgcolor = vec4f(0.1, 0.3, 0.6, 1.0); 
    const max_depth = 10; 
    var r = trace_rays(coords, uniforms); 
    var result = vec3f(0.0); 
    var hit: HitInfo; 
    for(var i = 0; i < max_depth; i++) { 
        if(intersect_scene(&r, &hit)) { result += shade(&r, &hit); } 
        else { result += bgcolor.rgb; break; } 
        if(hit.has_hit && !hit.continue_path) { break; } 
    } 
    // return vec4f(result, bgcolor.a); 
    return vec4f(pow(result, vec3f(1.0/uniforms.gamma)), bgcolor.a); 
} 