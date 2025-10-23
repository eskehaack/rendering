"use strict";
window.onload = function() { main(); }

function render(device, context, pipeline, bindGroup) {
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
        colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            loadOp: "clear",
            storeOp: "store",
        }]
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(4);
    pass.end();
    device.queue.submit([encoder.finish()]);
}

async function load_texture(device, filename) { 
    const response = await fetch(filename); 
    const blob = await response.blob(); 
    const img = await createImageBitmap(blob, { colorSpaceConversion: 'none' }); 
    const texture = device.createTexture({ 
        size: [img.width, img.height, 1], 
        format: "rgba8unorm", 
        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT 
    }); 
    device.queue.copyExternalImageToTexture( 
        { source: img, flipY: true }, 
        { texture: texture }, 
        { width: img.width, height: img.height }, 
    ); 
    return texture; 
}

async function main(){
    // Setup framework
    const gpu = navigator.gpu
    const adapter = await gpu.requestAdapter();
    const device = await adapter.requestDevice();
    const canvas = document.getElementById('webgpu-canvas');
    const context = canvas.getContext('webgpu');
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: canvasFormat,
    });

    // import wgsl file
    const wgslfile = document.getElementById('wgsl').src;
    const wgslcode = await fetch(wgslfile, {cache: "reload"}).then(r => r.text());
    const wgsl = device.createShaderModule({
        code: wgslcode
    });

    // Create pipline for drawing
    const pipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: { module: wgsl, entryPoint: 'main_vs', },
        fragment: {
        module: wgsl,
        entryPoint: 'main_fs',
        targets: [{ format: canvasFormat }], },
        primitive: { topology: 'triangle-strip', },
    });

    let bytelength = 6*sizeof['vec4']; // Buffers are allocated in vec4 chunks
    let uniforms = new ArrayBuffer(bytelength);
    const uniformBuffer = device.createBuffer({
        size: uniforms.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    let texture = await load_texture(device, "./grass.jpg")
    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            {
                binding: 0,
                resource: { buffer: uniformBuffer }
            },
            { 
                binding: 1,  
                resource: texture.createView() 
            },
        ],
    });

    const glassSelect = document.getElementById('glass-shader-select');
    const mattSelect = document.getElementById('matt-shader-select');
    const repeatSelect = document.getElementById('repeat-select');
    const filterSelect = document.getElementById('filtering-select');
    window.selectedShaders = {
        glass: parseInt(glassSelect.value),
        matt: parseInt(mattSelect.value),
        repeat: parseInt(repeatSelect.value),
        filter: parseInt(filterSelect.value)
    };
    glassSelect.addEventListener('change', () => {
        window.selectedShaders.glass = parseInt(glassSelect.value);
        if(window.updateShaders) window.updateShaders();
    });
    mattSelect.addEventListener('change', () => {
        window.selectedShaders.matt = parseInt(mattSelect.value);
        if(window.updateShaders) window.updateShaders();
    });
    repeatSelect.addEventListener('change', () => {
        window.selectedShaders.repeat = parseInt(repeatSelect.value);
        if(window.updateShaders) window.updateShaders();
    });
    filterSelect.addEventListener('change', () => {
        window.selectedShaders.filter = parseInt(filterSelect.value);
        if(window.updateShaders) window.updateShaders();
    });

    // calculate basis vectors
    const eye = vec3(2.0, 1.5, 2.0);
    const p   = vec3(0.0, 0.5, 0.0);
    const u   = vec3(0.0, 1.0, 0.0);

    const v = normalize(subtract(p, eye))
    const b1 = normalize(cross(v, u));
    const b2 = cross(b1, v);

    
    const aspect = canvas.width/canvas.height;
    var cam_const = 1.0;
    var gamma = 2.4
    var glass_shader = window.selectedShaders ? window.selectedShaders.glass : 0;
    var matt_shader = window.selectedShaders ? window.selectedShaders.matt : 0;
    var repeat_selector = window.selectedShaders ? window.selectedShaders.repeat : 0;
    var filter_selector = window.selectedShaders ? window.selectedShaders.filter : 0;
    new Float32Array(uniforms, 0, 24).set([
        aspect, cam_const, gamma, 0.0, 
        ...eye, 0.0,
        ...v, 0.0,
        ...b1, 0.0,
        ...b2, 0.0,
        glass_shader, matt_shader, repeat_selector, filter_selector
    ]);

    window.updateShaders = function() {
        glass_shader = window.selectedShaders.glass;
        matt_shader = window.selectedShaders.matt;
        repeat_selector = window.selectedShaders.repeat;
        filter_selector = window.selectedShaders.filter;
        let f32 = new Float32Array(uniforms);
        f32[20] = glass_shader;
        f32[21] = matt_shader;
        f32[22] = repeat_selector;
        f32[23] = filter_selector;
        requestAnimationFrame(animate);
    }

    device.queue.writeBuffer(uniformBuffer, 0, uniforms)
    addEventListener("wheel", (event) => {
        gamma *= 1.0 + 2.5e-4*event.deltaY;
        new Float32Array(uniforms, 8, 1).set([gamma]);
        requestAnimationFrame(animate);
    });
    function animate() {
        device.queue.writeBuffer(uniformBuffer, 0, uniforms);
        render(device, context, pipeline, bindGroup);
    }
    animate();
}