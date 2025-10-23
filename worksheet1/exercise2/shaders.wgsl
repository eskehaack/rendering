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

// Define Ray struct
struct Ray {
    origin: vec3f,
    direction: vec3f,
    tmin: f32,
    tmax: f32
}

@vertex
fn main_vs(@builtin(vertex_index) VertexIndex : u32) -> VSOut {
    const pos = array<vec2f, 4>(vec2f(-1.0, 1.0), vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(1.0, -1.0));
    var vsOut: VSOut;
    vsOut.position = vec4f(pos[VertexIndex], 0.0, 1.0);
    vsOut.coords = pos[VertexIndex];
    return vsOut;
}

fn trace_rays(ipcoords: vec2f, uniforms: Uniforms) -> Ray {
    let xip = ipcoords.x * uniforms.aspect * 0.5f;
    let yip = ipcoords.y * 0.5f;

    let qb1 = uniforms.b1 * xip;
    let qb2 = uniforms.b2 * yip;
    let qv = uniforms.v * uniforms.cam_const + 1.0;

    let origin = uniforms.eye;
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
    // Map coords to image plane
    var r = trace_rays(coords, uniforms);
    // ... use direction for ray tracing, shading, etc.
    return vec4f(r.direction, 1.0);
}