"use strict";
window.onload = function() { main(); }

function render(device, context, pipeline, bindGroup, timingHelper, gpuTime) {
    const encoder = device.createCommandEncoder();
    const pass = timingHelper.beginRenderPass(encoder, {
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
    timingHelper.getResult().then( time => { 
        gpuTime = time/1000; 
    });

    return gpuTime;
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

function compute_jitters(jitter, pixelsize, subdivs) { 
    const step = pixelsize/subdivs; 
    if(subdivs < 2) { 
        jitter[0] = 0.0; 
        jitter[1] = 0.0; 
    } else { 
        for(var i = 0; i < subdivs; ++i) {
            for(var j = 0; j < subdivs; ++j) { 
                const idx = (i*subdivs + j)*2; 
                jitter[idx] = (Math.random() + j)*step - pixelsize*0.5; 
                jitter[idx + 1] = (Math.random() + i)*step - pixelsize*0.5; 
            }
        }
    }
}

async function main(){
    // Setup framework
    const gpu = navigator.gpu
    const adapter = await gpu.requestAdapter();
    const canTimestamp = adapter.features.has('timestamp-query');
    console.log("Timestamp query support: ", canTimestamp);
    const device = await adapter.requestDevice({
        requiredFeatures: [
            ...(canTimestamp ? ['timestamp-query'] : []),
        ],
    })
    const timingHelper = new TimingHelper(device);
    let gpuTime = 0; 
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

    // import obj file
    const obj_filename = '../objects/bunny.obj';
    const obj = await readOBJFile(obj_filename, 1, true); // file name, scale, ccw vertices
    var bspBuffers = {};
    bspBuffers = build_bsp_tree(obj, device, bspBuffers);
    const positionBuffer = bspBuffers.positions;
    const indexBuffer = bspBuffers.indices;
    const normalBuffer = bspBuffers.normals;
    const aabbBuffer = bspBuffers.aabb;

    let mat_bytelength = obj.materials.length*2*sizeof['vec4'];
    var materials = new ArrayBuffer(mat_bytelength);
    const materialBuffer = device.createBuffer({
        size: mat_bytelength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    });
    for(var i = 0; i < obj.materials.length; ++i) {
        const mat = obj.materials[i];
        const emission = vec4(mat.emission.r, mat.emission.g, mat.emission.b, mat.emission.a);
        const color = vec4(mat.color.r, mat.color.g, mat.color.b, mat.color.a);
        new Float32Array(materials, i*2*sizeof['vec4'], 8).set([...emission, ...color]);
    }
    device.queue.writeBuffer(materialBuffer, 0, materials);
    const matidxBuffer = device.createBuffer({
        size: obj.mat_indices.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    });
    device.queue.writeBuffer(matidxBuffer, 0, obj.mat_indices);
    // Ensure we never create a zero-sized GPU buffer for bindings (validation requires min size)
    var lightIndicesArray = obj.light_indices;
    if (!(lightIndicesArray instanceof Uint32Array)) {
        lightIndicesArray = new Uint32Array(lightIndicesArray || []);
    }
    if (lightIndicesArray.byteLength === 0) {
        // fallback 1-element array so the GPU binding has a valid non-zero size
        lightIndicesArray = new Uint32Array([0]);
    }
    const lightIndexBuffer = device.createBuffer({
        size: lightIndicesArray.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
    });
    device.queue.writeBuffer(lightIndexBuffer, 0, lightIndicesArray);

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

    let bytelength = 7*sizeof['vec4']; // Buffers are allocated in vec4 chunks
    let uniforms = new ArrayBuffer(bytelength);
    const uniformBuffer = device.createBuffer({
        size: uniforms.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    let jitter = new Float32Array(200); // allowing subdivs from 1 to 10 
    const jitterBuffer = device.createBuffer({ 
        size: jitter.byteLength, 
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE 
    });
    const subdivSeletor = document.getElementById('subdiv-selector');
    const pixelSize = 1/canvas.height
    window.antiAlias = {
        subdiv: parseInt(subdivSeletor.value),
    }
    subdivSeletor.addEventListener('change', () => {
        window.antiAlias.subdiv = parseInt(subdivSeletor.value);
        compute_jitters(jitter, pixelSize, window.antiAlias.subdiv);
        let f32 = new Float32Array(uniforms);
        f32[25] = window.antiAlias.subdiv;
        device.queue.writeBuffer(uniformBuffer, 0, uniforms);
        device.queue.writeBuffer(jitterBuffer, 0, jitter);
        requestAnimationFrame(animate);
    });

    const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            {
                binding: 0,
                resource: { buffer: uniformBuffer }
            },
            { 
                binding: 1,  
                resource: { buffer: jitterBuffer}
            },
            {
                binding: 2,
                resource: { buffer: positionBuffer }
            },
            {
                binding: 3,
                resource: { buffer: indexBuffer }
            },
            {
                binding: 4,
                resource: { buffer: normalBuffer }
            },
            {
                binding: 5,
                resource: { buffer: aabbBuffer }
            },
            {
                binding: 6,
                resource: { buffer: materialBuffer }
            },
            {
                binding: 7,
                resource: { buffer: bspBuffers.bspTree }
            },
            {
                binding: 8,
                resource: { buffer: bspBuffers.bspPlanes }
            },
            {
                binding: 9,
                resource: { buffer: bspBuffers.treeIds }
            }
        ],
    });

    const mattSelect = document.getElementById('matt-shader-select');
    const repeatSelect = document.getElementById('repeat-select');
    const filterSelect = document.getElementById('filtering-select');
    const textureEnable = document.getElementById('texture-enable');
    const textureScaleInput = document.getElementById('texture-scale-input');
    window.selectedOptions = {
        matt: parseInt(mattSelect.value),
        repeat: parseInt(repeatSelect.value),
        filter: parseInt(filterSelect.value),
        texture: textureEnable.checked ? 1 : 0,
        textureScale: parseFloat(textureScaleInput.value)
    };
    mattSelect.addEventListener('change', () => {
        window.selectedOptions.matt = parseInt(mattSelect.value);
        if(window.updateOptions) window.updateOptions();
    });
    repeatSelect.addEventListener('change', () => {
        window.selectedOptions.repeat = parseInt(repeatSelect.value);
        if(window.updateOptions) window.updateOptions();
    });
    filterSelect.addEventListener('change', () => {
        window.selectedOptions.filter = parseInt(filterSelect.value);
        if(window.updateOptions) window.updateOptions();
    });
    textureEnable.addEventListener('change', () => {
        window.selectedOptions.texture = textureEnable.checked ? 1 : 0;
        if(window.updateOptions) window.updateOptions();
    });
    textureScaleInput.addEventListener('change', () => {
        window.selectedOptions.textureScale = parseFloat(textureScaleInput.value);
        if(window.updateOptions) window.updateOptions();
    });

    // calculate basis vectors
    const eye = vec3(-0.02, 0.11, 0.6);
    const p   = vec3(-0.02, 0.11, 0.0);
    const u   = vec3(0.0, 1.0, 0.0);

    const v = normalize(subtract(p, eye))
    const b1 = normalize(cross(v, u));
    const b2 = cross(b1, v);

    
    const aspect = canvas.width/canvas.height;
    var cam_const = 3.5;
    var gamma = 2.4;
    var matt_shader = window.selectedOptions ? window.selectedOptions.matt : 0;
    var repeat_selector = window.selectedOptions ? window.selectedOptions.repeat : 0;
    var filter_selector = window.selectedOptions ? window.selectedOptions.filter : 0;
    var texture_enable = window.selectedOptions ? window.selectedOptions.texture : 1;
    var subdivs = window.antiAlias ? window.antiAlias.subdiv : 1;
    var textureScale = window.selectedOptions ? window.selectedOptions.textureScale : 1;
    new Float32Array(uniforms, 0, 28).set([
        aspect, cam_const, gamma, 0.0, 
        ...eye, 0.0,
        ...v, 0.0,
        ...b1, 0.0,
        ...b2, 0.0,
        matt_shader, repeat_selector, filter_selector, 0.0, 
        texture_enable, subdivs, textureScale, 0.0
    ]);

    window.updateOptions = function() {
        matt_shader = window.selectedOptions.matt;
        repeat_selector = window.selectedOptions.repeat;
        filter_selector = window.selectedOptions.filter;
        texture_enable = window.selectedOptions.texture;
        subdivs = window.antiAlias.subdiv;
        textureScale = window.selectedOptions.textureScale;
        let f32 = new Float32Array(uniforms);
        f32[20] = matt_shader;
        f32[21] = repeat_selector;
        f32[22] = filter_selector;
        f32[24] = texture_enable;
        f32[25] = subdivs;
        f32[26] = textureScale;
        device.queue.writeBuffer(uniformBuffer, 0, uniforms);
        device.queue.writeBuffer(jitterBuffer, 0, jitter);
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
        device.queue.writeBuffer(jitterBuffer, 0, jitter);
        requestAnimationFrame(animate);
        gpuTime = render(device, context, pipeline, bindGroup, timingHelper, gpuTime);
    }
    animate();
    document.getElementById("stats").value = `Rendering time: ${gpuTime.toFixed(2)} ms`
}