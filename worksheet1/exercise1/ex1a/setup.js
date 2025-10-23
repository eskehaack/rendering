"use strict";
window.onload = function() { main(); }

async function main(){
    const canvas = document.querySelector("canvas");

    // WebGPU device initialization
    if (!navigator.gpu) {
        throw new Error("WebGPU not supported on this browser.");
        console.log('1');
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error("No appropriate GPUAdapter found.");
    }

    const device = await adapter.requestDevice();

    // Canvas configuration
    const context = canvas.getContext("webgpu");
    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: canvasFormat,
    });

    // Clear the canvas with a render pass
    const encoder = device.createCommandEncoder();

    const pass = encoder.beginRenderPass({
        colorAttachments: [{
                view: context.getCurrentTexture().createView(),
                loadOp: "clear",
                storeOp: "store",
            }]
    });

    pass.end();

    device.queue.submit([encoder.finish()]);
}

