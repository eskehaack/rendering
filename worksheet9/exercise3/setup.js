"use strict";
window.onload = function() { main(); }

async function render(device, context, pipeline, bindGroup, timingHelper, gpuTime, textures) {
    const encoder = device.createCommandEncoder();
    const pass = timingHelper.beginRenderPass(encoder, {
        colorAttachments: [
            {
                view: context.getCurrentTexture().createView(),
                loadOp: "clear",
                storeOp: "store",
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            },
            {
                view: textures.renderSrc.createView(),
                loadOp: "load",
                storeOp: "store",
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            }
        ]
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(4);
    pass.end();

    encoder.copyTextureToTexture(
        { texture: textures.renderSrc }, 
        { texture: textures.renderDst }, 
        [textures.width, textures.height]
    );

    device.queue.submit([encoder.finish()]);
    const time = await timingHelper.getResult();
    gpuTime = time / 1000;

    return gpuTime;
}

async function load_texture(device, filename) { 
// HDR-aware loader using HDRImage from hdrpng.js for .hdr / .rgbe images.
    if (filename.match(/\.hdr$/i) || filename.match(/\.rgbe$/i) || filename.match(/\.rgb9_e5\.png$/i) || filename.match(/\.hdr\.png$/i)) {
        return new Promise((resolve, reject) => {
            try {
                const hdr = new HDRImage();
                hdr.onload = () => {
                    const w = hdr.width;
                    const h = hdr.height;
                    const rgb = hdr.dataFloat;
                    const rgba = new Float32Array(w * 4); // row buffer (reused)
                    const texture = device.createTexture({
                        size: [w, h, 1],
                        format: 'rgba32float',
                        usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT
                    });

                    const bytesPerRow = 4 * 4 * w; // 4 components * 4 bytes * width
                    const alignedBytesPerRow = Math.ceil(bytesPerRow / 256) * 256;
                    const floatsPerRow = alignedBytesPerRow / 4;

                    const rowBuffer = new ArrayBuffer(alignedBytesPerRow);
                    const rowF32 = new Float32Array(rowBuffer);

                    for (let row = 0; row < h; ++row) {
                        const srcOffset = row * w * 3; // rgb has 3 floats per pixel
                        // fill rgba row (length w*4)
                        for (let i = 0, j = srcOffset; i < w; ++i, j += 3) {
                            const dst = i * 4;
                            rowF32[dst + 0] = rgb[j + 0];
                            rowF32[dst + 1] = rgb[j + 1];
                            rowF32[dst + 2] = rgb[j + 2];
                            rowF32[dst + 3] = 1.0;
                        }
                        for (let p = w * 4; p < floatsPerRow; ++p) rowF32[p] = 0.0;

                        device.queue.writeTexture(
                            { texture: texture, origin: { x: 0, y: row, z: 0 } },
                            rowBuffer,
                            { bytesPerRow: alignedBytesPerRow, rowsPerImage: 1 },
                            { width: w, height: 1, depthOrArrayLayers: 1 }
                        );
                    }

                    resolve(texture);
                };
                hdr.onerror = (e) => reject(e);
                hdr.src = filename;
                hdr.gamma = 2.2;
            } catch (e) {
                reject(e);
            }
        });
    } else {
        // fallback LDR loader (existing path)
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
}

function compute_jitters(jitter, pixelsize, subdivs) { 

    if (!subdivs || subdivs <= 1) {
        // no subpixel jitter -> keep all zeros
        return;
    }

    let idx = 0;
    const step = pixelsize/subdivs; 
    for(var i = 0; i < subdivs; ++i) {
        for(var j = 0; j < subdivs; ++j) { 
            jitter[idx++] = 2.0 * (Math.random() + j) * step - pixelsize * 0.5; 
            jitter[idx++] = 2.0 * (Math.random() + i) * step - pixelsize * 0.5; 
            jitter[idx++] = 0.0;
            jitter[idx++] = 0.0;
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

    let textures = new Object();
    textures.width = canvas.width;
    textures.height = canvas.height;
    textures.renderSrc = device.createTexture({
        size: [canvas.width, canvas.height],
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        format: 'rgba32float',
    });
    textures.renderDst = device.createTexture({
        size: [canvas.width, canvas.height],
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
        format: 'rgba32float',
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
        colorAttachments: [{
            view: textures.renderDst.createView(),
            loadOp: 'clear',
            clearValue: [0.0, 0.0, 0.0, 0.0],
            storeOp: 'store',
        }]
    });
    pass.end();
    device.queue.submit([enc.finish()]);

    // import obj file
    const obj_filename = '../../helper_functions/objects/teapot.obj';
    const obj = await readOBJFile(obj_filename, 1, true); // file name, scale, ccw vertices
    var bspBuffers = {};
    bspBuffers = build_bsp_tree(obj, device, bspBuffers);

    let texture = await load_texture(device, "./../../helper_functions/luxo_pxr_campus.jpg")

    let mat_bytelength = obj.materials.length*2*sizeof['vec4'];
    const materialBuffer = device.createBuffer({
        size: mat_bytelength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    });
    // Populate and upload material data (emission + diffuse) so shaders can read proper colors
    var materials = new ArrayBuffer(mat_bytelength);
    for(var i = 0; i < obj.materials.length; ++i) {
        const mat = obj.materials[i];
        const emission = mat.emission ? vec4(mat.emission.r, mat.emission.g, mat.emission.b, mat.emission.a) : vec4(0.0,0.0,0.0,0.0);
        const color = mat.color ? vec4(mat.color.r, mat.color.g, mat.color.b, mat.color.a) : vec4(0.8,0.8,0.8,1.0);
        new Float32Array(materials, i*2*sizeof['vec4'], 8).set([...emission, ...color]);
    }
    device.queue.writeBuffer(materialBuffer, 0, materials);
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
            targets: [{ format: canvasFormat }, { format: "rgba32float" }]
        },
        primitive: { topology: 'triangle-strip', },
    });

    const MAX_JITTERS = 100; // 100 vec2 -> padded to 100 vec4
    let jitter = new Float32Array(MAX_JITTERS * 4); // xy = jitter, zw = padding
    const subdivSeletor = document.getElementById('subdiv-selector');
    const pixelSize = 1/canvas.height
    window.antiAlias = {
        subdiv: parseInt(subdivSeletor.value),
    }
    // compute initial jitters and upload so shader has them on first frame
    compute_jitters(jitter, pixelSize, window.antiAlias.subdiv);

    subdivSeletor.addEventListener('change', () => {
        window.antiAlias.subdiv = parseInt(subdivSeletor.value);
        compute_jitters(jitter, pixelSize, window.antiAlias.subdiv);
        let f32 = new Float32Array(uniforms);
        f32[25] = window.antiAlias.subdiv;
        if (window.antiAlias.subdiv <= 0) {
            f32[25] = 1; // avoid zero subdivs in the shader
        } else {
            f32[15] = window.antiAlias.subdiv * window.antiAlias.subdiv; // no_of_jitters
        }
        new Float32Array(uniforms, 7 * sizeof['vec4'], jitter.length).set(jitter);
        device.queue.writeBuffer(uniformBuffer, 0, uniforms);
        requestAnimationFrame(animate);
    });

    let bytelength = 7*sizeof['vec4'] + jitter.byteLength; // Buffers are allocated in vec4 chunks
    let uniforms = new ArrayBuffer(bytelength);
    const uniformBuffer = device.createBuffer({
        size: uniforms.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
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
                resource: textures.renderDst.createView()
            },
            {
                binding: 2,
                resource: { buffer: bspBuffers.attribs }
            },
            {
                binding: 3,
                resource: { buffer: bspBuffers.indices }
            },
            {
                binding: 4,
                resource: { buffer: bspBuffers.aabb }
            },
            {
                binding: 5,
                resource: texture.createView() 
            },
            {
                binding: 6,
                resource: { buffer: bspBuffers.bspTree }
            },
            {
                binding: 7,
                resource: { buffer: bspBuffers.bspPlanes }
            },
            {
                binding: 8,
                resource: { buffer: bspBuffers.treeIds }
            },
            {
                binding: 9,
                resource: { buffer: lightIndexBuffer }
            },
            {
                binding: 10,
                resource: { buffer: materialBuffer }
            },
        ],
    });
    function resetAccumulation() {
        // reset frame counter to 0 so accumulation restarts cleanly
        let u32 = new Uint32Array(uniforms);
        u32[11] = 0;

        // upload uniforms update
        device.queue.writeBuffer(uniformBuffer, 0, uniforms);

        // clear the accumulation texture (renderDst) so we start from zeros
        const enc = device.createCommandEncoder();
        const clearPass = enc.beginRenderPass({
            colorAttachments: [{
                view: textures.renderDst.createView(),
                loadOp: 'clear',
                clearValue: [0.0, 0.0, 0.0, 0.0],
                storeOp: 'store',
            }]
        });
        clearPass.end();
        device.queue.submit([enc.finish()]);
        requestAnimationFrame(animate);
    }
    const mattSelect = document.getElementById('matt-shader-select');
    const repeatSelect = document.getElementById('repeat-select');
    const filterSelect = document.getElementById('filtering-select');
    const textureEnable = document.getElementById('texture-enable');
    const textureScaleInput = document.getElementById('texture-scale-input');
    const bgCheckbox = document.getElementById('use-bg-checkbox');
    const progCheckbox = document.getElementById('progressive-enable');
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
        resetAccumulation();
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
        resetAccumulation();
    });
    progCheckbox.addEventListener('change', async () => {
        // patch the progressive flag (float slot index 27)
        let f32 = new Float32Array(uniforms);
        f32[27] = progCheckbox.checked ? 1.0 : 0.0;
        resetAccumulation();
    });
    // Blue background checkbox handling (use_bg stored at float index 28)
    bgCheckbox.addEventListener('change', async () => {
        let f32 = new Float32Array(uniforms);
        f32[23] = bgCheckbox.checked ? 1.0 : 0.0;
        resetAccumulation();
    });

    // calculate basis vectors
    const eye = vec3(0.15, 1.5, 10.0);
    const p   = vec3(0.15, 1.5, 0.0);
    const u   = vec3(0.0, 1.0, 0.0);

    const v = normalize(subtract(p, eye))
    const b1 = normalize(cross(v, u));
    const b2 = cross(b1, v);

    
    const aspect = canvas.width/canvas.height;
    var cam_const = 1.0;
    var gamma = 1.5;
    var frame = 0;
    var no_of_jitters = window.antiAlias.subdiv * window.antiAlias.subdiv;
    var matt_shader = window.selectedOptions ? window.selectedOptions.matt : 0;
    var repeat_selector = window.selectedOptions ? window.selectedOptions.repeat : 0;
    var filter_selector = window.selectedOptions ? window.selectedOptions.filter : 0;
    var texture_enable = window.selectedOptions ? window.selectedOptions.texture : 1;
    var subdivs = window.antiAlias ? window.antiAlias.subdiv : 1;
    var textureScale = window.selectedOptions ? window.selectedOptions.textureScale : 1;
    var progressive = progCheckbox.checked ? 1.0 : 0.0;
    var bgColor = bgCheckbox.checked ? 1.0 : 0.0;
    new Float32Array(uniforms, 0, 28).set([
        aspect, cam_const, textures.width, textures.height,
        ...eye, gamma,
        ...v, frame,
        ...b1, no_of_jitters,
        ...b2, 0.0,
        matt_shader, repeat_selector, filter_selector, bgColor,
        texture_enable, subdivs, textureScale, progressive
    ]);
    new Float32Array(uniforms, 7 * sizeof['vec4'], jitter.length).set(jitter);

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
        requestAnimationFrame(animate);
    }
    // helper to increment the frame counter stored at element index 11
    function bumpFrameAndUpload(device, uniformBuffer, uniforms) {
        // uniforms is the ArrayBuffer used earlier with Float32Array(...).set([...])
        const u32 = new Uint32Array(uniforms);
        u32[11] = (u32[11] || 0) + 1; // increment frame (stored as u32 in the shader)
        console.log("Frame:", u32[11]);
        device.queue.writeBuffer(uniformBuffer, 0, uniforms);
        return u32[11];
    };
    device.queue.writeBuffer(uniformBuffer, 0, uniforms);
    async function animate() {
        device.queue.writeBuffer(uniformBuffer, 0, uniforms);
        (async () => {
            gpuTime = await render(device, context, pipeline, bindGroup, timingHelper, gpuTime, textures);
            document.getElementById("stats").value = `Rendering time: ${gpuTime.toFixed(2)} μs`;
        })();
        var frame = bumpFrameAndUpload(device, uniformBuffer, uniforms);
        if (progCheckbox.checked && frame < 50) {
            requestAnimationFrame(animate);
        }
    }
    animate();
}