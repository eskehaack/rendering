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
    glass_shader: f32,
    matt_shader: f32,
    repeat_selector: f32,
    filter_selector: f32,
    texture_enable: f32,
    _pad5: f32,
    _pad6: f32,
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
    ambient: vec3f,
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

// Bindings from JS

// Basic info
@group(0) @binding(0) var<uniform> uniforms : Uniforms;
// Texture
@group(0) @binding(1) var texture: texture_2d<f32>;

// Intersection functions
fn intersect_plane(r: Ray, hit: ptr<function, HitInfo>, position: vec3f, onb: Onb) -> bool {
    let t = -dot(r.origin, onb.normal) / dot(r.direction, onb.normal);
    let p = r.origin + r.direction * t;
    let has_hit = r.tmin < t && t < r.tmax;
    if has_hit {
        (*hit).ambient = 0.1 * (*hit).color ;
        (*hit).diffuse = 0.9 * (*hit).color ;
        (*hit).specular = vec3f(0.0, 0.0, 0.0);

        (*hit).position = p;
        (*hit).dist = t; 
        (*hit).normal = normalize(onb.normal);
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
    (*hit).dist = t;
    (*hit).position = r.origin + r.direction * t;
    (*hit).normal = normalize((*hit).position - center);
    (*hit).has_hit = true;

    (*hit).shader = i32(uniforms.glass_shader);

    return true;
}
fn intersect_scene(r: ptr<function, Ray>, hit: ptr<function, HitInfo>) -> bool {
    var has_hit = false;
    (*hit).has_hit = false;

    // plane
    let plane_position = vec3f(0.0, 0.0, 0.0);
    let plane_onb = Onb(vec3f(-1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), vec3f(0.0, 1.0, 0.0));
    if intersect_plane(*r, hit, plane_position, plane_onb) {
        let x = hit.position;
        let x0 = plane_position;
        let u = dot(x - x0, plane_onb.tangent);
        let v = dot(x - x0, plane_onb.binormal);
        let texcoords = 0.2 * vec2f(u, v);
        if (uniforms.texture_enable == 1.0) {
            (*hit).color = texture_switch(texture, texcoords);
        } else {
            (*hit).color = vec3f(0.1, 0.7, 0.0);
        }
    }

    // triangle
    let triangle_v = array<vec3f, 3>(
        vec3f(-0.2,0.1,0.9), 
        vec3f(0.2,0.1,0.9), 
        vec3f(-0.2,0.1,-0.1)
    );
    if intersect_triangle(*r, hit, triangle_v) {
        (*hit).color = vec3f(0.4,0.3,0.2);
    }

    // sphere
    let sphere_center = vec3f(0.0, 0.5, 0.0);
    let sphere_radius = 0.3;
    if intersect_sphere(*r, hit, sphere_center, sphere_radius) {
        (*hit).color = vec3f(0.0, 0.0, 0.0);
    };

    if hit.has_hit {
        (*hit).ambient = 0.1 * (*hit).color ;
        (*hit).diffuse = 0.9 * (*hit).color ;
        (*hit).specular = vec3f(0.1, 0.1, 0.1);
    };

    return hit.has_hit; 
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

// Shade functions
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

    let Lr = (hit.ambient + diffuse + specular) * Li * nwi;
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
        case 1: { return lambertian(r, hit); } 
        case 2: { return mirror(r, hit); } 
        case 3: { return refract_shader(r, hit); }
        case 4: { return phong(r, hit); }
        case 5: { return glossy(r, hit); }
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

    return vec4f(pow(result, vec3f(1.0/uniforms.gamma)), bgcolor.a); 
} 