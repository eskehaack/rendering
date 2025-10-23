struct VSOut {
    @builtin(position) position: vec4f,
    @location(0) coords : vec2f,
};

struct Uniforms {
    aspect: f32,
    cam_const: f32,
    eye: vec3f,
    v: vec3f,
    b1: vec3f,
    b2: vec3f
};
@group(0) @binding(0) var<uniform> uniforms : Uniforms;

struct HitInfo { 
    has_hit: bool, 
    dist: f32, 
    position: vec3f, 
    normal: vec3f, 
    color: vec3f, 
    shader: u32, 
}; 

// Define Ray struct
struct Ray {
    origin: vec3f,
    direction: vec3f,
    tmin: f32,
    tmax: f32
}

fn intersect_plane(r: Ray, hit: ptr<function, HitInfo>, position: vec3f, normal: vec3f) -> bool {
    let t = -dot(r.origin, normal) / dot(r.direction, normal);
    let p = r.origin + r.direction * t;
    let has_hit = r.tmin < t && t < r.tmax;
    if has_hit {
        (*hit).position = p;
        (*hit).dist = t; 
        (*hit).has_hit = true;
        (*hit).color = vec3f(0.1, 0.7, 0.0);
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

    if beta >= 0 && gamma >= 0 && (beta + gamma) <= 1 && t < (*hit).dist{
        (*hit).normal = normalize(n);
        (*hit).has_hit = true;
        (*hit).position = r.origin + r.direction * (*hit).dist;
        (*hit).color = vec3f(0.4,0.3,0.2);
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
    (*hit).dist = t;
    (*hit).position = r.origin + r.direction * t;
    (*hit).has_hit = true;
    (*hit).color = vec3f(0.0, 0.0, 0.0);
    return true;
}

fn trace_rays(ipcoords: vec2f, uniforms: Uniforms) -> Ray {
    let xip = ipcoords.x * uniforms.aspect * 0.5;
    let yip = ipcoords.y * 0.5;

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
    // Map coords to image plane
    var hit: HitInfo;
    var r = trace_rays(coords, uniforms);

    var has_hit: bool;
    has_hit = intersect_plane(r, &hit, vec3f(0.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0));
    has_hit = intersect_triangle(r, &hit, array<vec3f, 3>(vec3f(-0.2,0.1,0.9), vec3f(0.2,0.1,0.9), vec3f(-0.2,0.1,-0.1)));
    has_hit = intersect_sphere(r, &hit, vec3f(0.0, 0.5, 0.0), 0.3);
    if ! hit.has_hit{
        return vec4f(0.1,0.3,0.6,1.0);
    } else {
        return vec4f(hit.color, 1.0);
    };
}