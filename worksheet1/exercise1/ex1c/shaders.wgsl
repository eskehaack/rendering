struct VSOut {
    @builtin(position) position: vec4f,
    @location(0) coords : vec2f,
};

@vertex
fn main_vs(@builtin(vertex_index) VertexIndex : u32) -> VSOut {
    const pos = array<vec2f, 4>(vec2f(-1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(1.0, -1.0));
    var vsOut: VSOut;
    vsOut.position = vec4f(pos[VertexIndex], 0.0, 1.0);
    vsOut.coords = pos[VertexIndex];
    return vsOut;
}
// Define Ray struct
struct Ray {
    origin: vec3f,
    direction: vec3f,
    tmin: f32,
    tmax: f32
}

fn get_camera_ray(ipcoords: vec2f) -> Ray {
// Implement ray generation (WGSL has vector operations like normalize and cross)
    let xip = ipcoords.x;
    let yip = ipcoords.y;

    let eye = vec3f(2.0, 1.5, 2.0);
    let p   = vec3f(0.0, 0.5, 0.0);
    let u  = vec3f(0.0, 1.0, 0.0);

    let v = normalize(p - eye);
    let b1 = normalize(cross(v, u));
    let b2 = cross(b1, v);

    let qb1 = b1 * xip;
    let qb2 = b2 * yip;
    let qv = v;

    let origin = eye;
    let direction = normalize((qb1 + qb2 + qv));

    return Ray(
        origin,
        direction,
        0.0,
        10.0 // arbitrary
    );
}
@fragment
fn main_fs(@location(0) coords: vec2f) -> @location(0) vec4f {
    let ipcoords = coords*0.5;
    var r = get_camera_ray(ipcoords);
    return vec4f(r.direction*0.5 + 0.5, 1.0);
}