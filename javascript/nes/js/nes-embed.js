var SCREEN_WIDTH = 256;
var SCREEN_HEIGHT = 240;
var FRAMEBUFFER_SIZE = SCREEN_WIDTH*SCREEN_HEIGHT;

var canvas_ctx, image;
var framebuffer_u8, framebuffer_u32;

var AUDIO_BUFFERING = 512;
var SAMPLE_COUNT = 4*1024;
var SAMPLE_MASK = SAMPLE_COUNT - 1;
var audio_samples_L = new Float32Array(SAMPLE_COUNT);
var audio_samples_R = new Float32Array(SAMPLE_COUNT);
var audio_write_cursor = 0, audio_read_cursor = 0;
var audio_started = false;
var audioEnabled = true;

var nes;
var audio_ctx;
var script_processor;
var animationFrameId;

// Initialize or reset NES instance
function initNesInstance() {
	// Cancel any existing animation frame request
	if (animationFrameId) {
		window.cancelAnimationFrame(animationFrameId);
		animationFrameId = null;
	}
	
	// Clear audio buffers
	audio_samples_L.fill(0);
	audio_samples_R.fill(0);
	audio_write_cursor = 0;
	audio_read_cursor = 0;
	
	// Create new NES instance
	nes = new jsnes.NES({
		onFrame: function(framebuffer_24){
			for(var i = 0; i < FRAMEBUFFER_SIZE; i++) framebuffer_u32[i] = 0xFF000000 | framebuffer_24[i];
		},
		onAudioSample: function(l, r){
			if(audioEnabled && audio_started){
				audio_samples_L[audio_write_cursor] = l;
				audio_samples_R[audio_write_cursor] = r;
			} else {
				audio_samples_L[audio_write_cursor] = 0;
				audio_samples_R[audio_write_cursor] = 0;
			}
			audio_write_cursor = (audio_write_cursor + 1) & SAMPLE_MASK;
		},
	});
}

function onAnimationFrame(){
	animationFrameId = window.requestAnimationFrame(onAnimationFrame);
	if (!audio_started) {
		nes.frame();
	}	

	image.data.set(framebuffer_u8);
	canvas_ctx.putImageData(image, 0, 0);
}

function audio_remain(){
	return (audio_write_cursor - audio_read_cursor) & SAMPLE_MASK;
}

function audio_callback(event){
	var dst = event.outputBuffer;
	var len = dst.length;
	
	// Attempt to avoid buffer underruns.
	if(audio_remain() < AUDIO_BUFFERING) nes.frame();
	
	var dst_l = dst.getChannelData(0);
	var dst_r = dst.getChannelData(1);
	for(var i = 0; i < len; i++){
		var src_idx = (audio_read_cursor + i) & SAMPLE_MASK;
		dst_l[i] = audio_samples_L[src_idx];
		dst_r[i] = audio_samples_R[src_idx];
	}
	
	audio_read_cursor = (audio_read_cursor + len) & SAMPLE_MASK;
}

function keyboard(callback, event){
    event.preventDefault();
	var player = 1;
	switch(event.keyCode){
		case 38: // UP
			callback(player, jsnes.Controller.BUTTON_UP); break;
		case 40: // Down
			callback(player, jsnes.Controller.BUTTON_DOWN); break;
		case 37: // Left
			callback(player, jsnes.Controller.BUTTON_LEFT); break;
		case 39: // Right
			callback(player, jsnes.Controller.BUTTON_RIGHT); break;
		case 65: // 'a' - qwerty, dvorak
		case 81: // 'q' - azerty
			callback(player, jsnes.Controller.BUTTON_A); break;
		case 83: // 's' - qwerty, azerty
		case 79: // 'o' - dvorak
			callback(player, jsnes.Controller.BUTTON_B); break;
		case 9: // Tab
			callback(player, jsnes.Controller.BUTTON_SELECT); break;
		case 13: // Return
			callback(player, jsnes.Controller.BUTTON_START); break;
		default: break;
	}
}

function nes_init(canvas_id){
	var canvas = document.getElementById(canvas_id);
	canvas_ctx = canvas.getContext("2d");
	image = canvas_ctx.getImageData(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);
	
	canvas_ctx.fillStyle = "black";
	canvas_ctx.fillRect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);
	
	// Allocate framebuffer array.
	var buffer = new ArrayBuffer(image.data.length);
	framebuffer_u8 = new Uint8ClampedArray(buffer);
	framebuffer_u32 = new Uint32Array(buffer);
	
	// Setup or reset audio context
	if (!audio_ctx) {
		audio_ctx = new window.AudioContext();
	} else if (audio_ctx.state === 'closed') {
		audio_ctx = new window.AudioContext();
	}
	
	// Setup or reset script processor
	if (script_processor) {
		script_processor.disconnect();
		script_processor.onaudioprocess = null;
	}
	
	script_processor = audio_ctx.createScriptProcessor(AUDIO_BUFFERING, 0, 2);
	script_processor.onaudioprocess = audio_callback;
	script_processor.connect(audio_ctx.destination);
	
	// Initialize NES instance
	initNesInstance();
	
	if (audio_ctx.state === 'suspended') {
		var audioButton = document.getElementById('audio');
		audioButton.style.display = 'block';
	} else {
		audio_started = true;
	}
}

function nes_boot(rom_data){
	nes.loadROM(rom_data);
	window.requestAnimationFrame(onAnimationFrame);
	
	// Notify that game is loaded
	if(window.nes_on_load){
		window.nes_on_load();
	}
}

function nes_load_data(canvas_id, rom_data){
	nes_init(canvas_id);
	nes_boot(rom_data);
}

var currentRomPath = null;
var currentRomData = null;

function nes_load_url(canvas_id, path){
	nes_init(canvas_id);
	currentRomPath = path;
	
	var req = new XMLHttpRequest();
	req.open("GET", path);
	req.overrideMimeType("text/plain; charset=x-user-defined");
	req.onerror = () => console.log(`Error loading ${path}: ${req.statusText}`);
	
	req.onload = function() {
		if (this.status === 200) {
			currentRomData = this.responseText;
			nes_boot(currentRomData);
		} else if (this.status === 0) {
			// Aborted, so ignore error
		} else {
			req.onerror();
		}
	};
	
	req.send();
}

function nes_reset(){
	if(currentRomData){
		nes.loadROM(currentRomData);
		if(window.nes_on_load){
			window.nes_on_load();
		}
	}
}

document.addEventListener('keydown', (event) => {keyboard(nes.buttonDown, event)});
document.addEventListener('keyup', (event) => {keyboard(nes.buttonUp, event)});

document.addEventListener('DOMContentLoaded', (event) => {
    var audioButton = document.getElementById('audio');
    audioButton.addEventListener('click', () => {
        audio_ctx.resume().then(() => {
			audio_started = true;
		});
        audioButton.style.display = 'none';
    });
});