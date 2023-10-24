import * as THREE from '../libs/build/three.module.js';
import {GPUComputationRenderer} from '../libs/examples/jsm/misc/GPUComputationRenderer.js';
import noise from './noise.js';

const ParticleEngine = (renderer, scene, camera, _options) => {
	const options = {
		texture_width: 64,
		texture_height: 64
	};

	Object.assign(options, _options);

	const computeVelShader = `
			
		uniform float uTime;
		uniform sampler2D textureRandom;
		uniform float uNoiseScale;
		uniform float uMaxRadius;
		uniform vec3 uHit;
		uniform float uIsMouseDown;
		uniform float uSpeed;

		${noise}

		const float PI = 3.141592653;

		vec2 rotate(vec2 v, float a) {
			float s = sin(a);
			float c = cos(a);
			mat2 m = mat2(c, -s, s, c);
			return m * v;
		}

		void main() {
		    vec2 uv = gl_FragCoord.xy / resolution.xy;
		    
		    vec3 pos = texture2D( texturePosition, uv ).xyz;
		    vec3 vel = texture2D( textureVelocity, uv ).xyz;
		    vec3 extra = texture2D( textureRandom, uv ).xyz;
			
			/*float posOffset = mix(extra.r, 1.0, .75) * .25;
			vec3 acc        = curlNoise(pos * posOffset + uTime * .3);
			acc.y += .2;
			acc.y *= 0.5;
			acc.xz *= 0.2;
			
			float speedOffset = mix(extra.b, 1.0, .75);
			vel += acc * .0005 * speedOffset;


			float d = length(pos.xz);

			float t = smoothstep(0.25, 1.0, d);

			// vec2 dir = normalize(pos.xz);
			// vec2 dirRotate = rotate(dir, PI * 0.65 * ( 1.0 + extra.g * 0.25));
			// vel.xy += dirRotate * 0.00035 * t;

			
			// if(d > uMaxRadius) {
			// 	float f = pow(2.0, (d - uMaxRadius)) * 0.01;
			// 	vel.xz -= f * dir * speedOffset;
			// }

			// vel.xz += dirRotate * 0.0002;
			*/
			const float decrease = .97;
			vel *= decrease;

			if(pos.x > 1.4){
				vel *= 0.0;
			}

			if(pos.y > 1.4){
				vel *= 0.0;
			}
		    gl_FragColor = vec4( vel, 1.0 );
		}
	`;

	const computePosShader =  `
			
		void main() {
		    vec2 uv = gl_FragCoord.xy / resolution.xy;
		    //vec2 uv = vUv;

		    vec3 pos = texture2D( texturePosition, uv ).xyz;
		    vec3 vel = texture2D( textureVelocity, uv ).xyz;

		    pos += vel;
		    if(pos.x> 1.4) pos.x = -1.4;

		    if(pos.y> 1.4) pos.y = -1.4;
		    gl_FragColor = vec4( pos, 1.0 );
		}
	`;

	const {texture_width} = options;
	const {texture_height} = options;

	const AMOUNT = texture_width * texture_height;
	const seed = Math.random() * 0xff;

	let time = Math.random() * 0xFF;
	let gpuCompute;

	let variables = {
    	TEXTURE_WIDTH:texture_width,
    	TEXTURE_HEIGHT:texture_height,
    	AMOUNT:AMOUNT
    }

    const computeVars = {

    }

	const rttTextures = {

	}

	let randomTexture;
	let posUniforms, velUniforms, particleUniforms;
	let maxSpeed = options.touch ? 60 / 90 : 1;
	let speedOffset = maxSpeed;
	let oldRtt;
	let material, mesh;
	let cnt = 0;

	let _shader, _shader2;

	const copyMaterial = new THREE.ShaderMaterial({
        uniforms: {
            resolution: { type: 'v2', value: new THREE.Vector2( texture_width, texture_height ) },
            uTexture: { type: 't', value: null }
        },
        vertexShader: `
        	varying vec2 vUv;

			void main() {
			    gl_Position = vec4( position, 1.0 );
			    vUv = uv;
			}
        `,
        fragmentShader: `
        	varying vec2 vUv;

			uniform vec2 resolution;
			uniform sampler2D uTexture;

			void main() {
			    vec2 uv = gl_FragCoord.xy / resolution.xy;
			    //vec2 uv = vUv;

			    vec3 color = texture2D( uTexture, uv ).xyz;
			    gl_FragColor = vec4( color, 1.0 );
			}
        `
    });

    const fboScene = new THREE.Scene();
    const fboCamera = new THREE.Camera();

    fboCamera.position.z = 1;

    const fboMesh = new THREE.Mesh( new THREE.PlaneBufferGeometry( 2, 2 ), copyMaterial );
    fboScene.add( fboMesh );

    const copyTexture = (input, output) => {
        _fboMesh.material = copyMaterial;
        _copyShader.uniforms.uTexture.value = input.texture;
        
        renderer.setRenderTarget(output);
        renderer.render( _fboScene, _fboCamera );
        renderer.setRenderTarget(null);
    };

	const createTexture = () => {
        let texture = new THREE.DataTexture( new Float32Array( AMOUNT * 4 ), texture_width, texture_height, THREE.RGBAFormat, THREE.FloatType );
        texture.minFilter = THREE.NearestFilter;
        texture.magFilter = THREE.NearestFilter;
        texture.needsUpdate = true;
        texture.generateMipmaps = false;
        texture.flipY = false;
        
        return texture;
    };

	const initGPUCompute = _ => {
		console.log('application initGPUCompute')
		const rttSize = texture_width
		console.log(`rttSize:${rttSize}`)

		gpuCompute = new GPUComputationRenderer(rttSize, rttSize, renderer);

		rttTextures['positionTexture'] = gpuCompute.createTexture();
 		rttTextures['velocityTexture'] = gpuCompute.createTexture();

 		initPositionTexture(rttTextures['positionTexture'], 1)
 		initVelocityTexture(rttTextures['velocityTexture'])

 		randomTexture = createTexture();
 		initRandomTexture(randomTexture)

 		rttTextures['positionTexture'].needsUpdate = true;

 		const comPosition = computeVars['comPosition'] = gpuCompute.addVariable('texturePosition', computePosShader, rttTextures['positionTexture'])
    	const comVelocity = computeVars['comVelocity'] = gpuCompute.addVariable('textureVelocity', computeVelShader, rttTextures['velocityTexture'])

    	gpuCompute.setVariableDependencies(comPosition, [comPosition, comVelocity])

    	oldRtt = rttTextures['positionTexture'];

    	posUniforms = comPosition.material.uniforms
    	
    	gpuCompute.setVariableDependencies(comVelocity, [comVelocity, comPosition])
	    
	    velUniforms = comVelocity.material.uniforms
	    velUniforms.uTime = { type: 'f' , value:time }
	    velUniforms.textureRandom = { type: 'f' , value: randomTexture	 }
	    velUniforms.uHit =  { type:'v3', value:new THREE.Vector3() }
		velUniforms.uIsMouseDown =  { type:'f', value:0.0 }
		velUniforms.uNoiseScale = { type:'f', value:0.05}
		//velUniforms.uMinRadius = { type:'f', value:2.0}
		velUniforms.uMaxRadius = { type:'f', value:2.0}
		velUniforms.uSpeed = { type:'f', value:1.8 }
	    //initDepthRTT()
	    gpuCompute.init()
	};

	const initPositionTexture = (texture, mode = 0) => {
		const data = texture.image.data
		//console.log(mesh.geometry.attributes.position.count, variables.AMOUNT)
		const rand = THREE.MathUtils.randFloat;

		for(let i = 0; i < AMOUNT; i++){
			const radius = 1;//rand(1,1)
      		const phi = (Math.random() - 0.5) * Math.PI
      		const theta = Math.random() * Math.PI * 2

      		if(mode == 0){ // SPHERE
      			data[i * 4] = radius * Math.cos(theta) * Math.cos(phi)
		      	data[i * 4 + 1] = radius * Math.sin(phi)
		      	data[i * 4 + 2] = radius * Math.sin(theta) * Math.cos(phi)
		      	data[i * 4 + 3] = rand(0., 1.0)
      		}else if(mode == 1){  // CUBE   			
				data[i * 4] = rand(-1.4, 1.4)
				data[i * 4 + 1] = rand(-1.4, 1.4)
				data[i * 4 + 2] = rand(-1.4, 1.4)
				data[i * 4 + 3] = rand(0., 1.0)
      		}else if(mode == 2){ // PLANE
      			const DIM = 130

      			data[i * 4 + 0] = DIM / 2 - DIM * (i % texture_width) / texture_width
				data[i * 4 + 1] = DIM / 2 - DIM * ~~(i / texture_width) / texture_height
				data[i * 4 + 2] = 0	
				data[i * 4 + 3] = rand(0., 1.0)
      		}else{
      			console.log(mesh.geometry.attributes.position.count)
      		}
		}

		texture.needsUpdate = true;
	};

	const initVelocityTexture = texture => {
		const data = texture.image.data;
		const rand = THREE.MathUtils.randFloat;

		for(let i=0; i<AMOUNT; i++){
			data[i * 4] = rand(0.005, 0.01);//rand(-1.0, 1.0)
			data[i * 4 + 1] = 0.;//rand(-1.0, 1.0)
			data[i * 4 + 2] = 0.;//rand(-1.0, 1.0)
			data[i * 4 + 3] = 1.;//1.
		}
		
		texture.needsUpdate = true;
	};

	const initRandomTexture = texture => {
		const data = texture.image.data;
		const rand = THREE.MathUtils.randFloat;

		for(let i=0; i<AMOUNT; i++){
			data[i * 4] = rand(0, 1)
			data[i * 4 + 1] = rand(0, 1)
			data[i * 4 + 2] = rand(0, 1)
			data[i * 4 + 3] = Math.random()
		}

		texture.needsUpdate = true;
	};

	const update = (delta, time, mouse3d, isMouseDown) => {
		gpuCompute.compute();
		rttTextures['positionTexture'].needsUpdate = true;

		let f = isMouseDown ? 10.0 : 0.0;
		const r = isMouseDown ? 8.0 : 2.0;

		velUniforms.uTime.value += .01;
		velUniforms.uHit.value.copy(mouse3d);
		velUniforms.uIsMouseDown.value = f;
		//console.log(velUniforms.uTime.value);

		const comPosition = computeVars['comPosition']
		const comVelocity = computeVars['comVelocity'] 
	};

	initGPUCompute();

	const base = {
		update,
	};

	Object.defineProperty(base, 'texturePosition', {
		get:() => gpuCompute.getCurrentRenderTarget(computeVars['comPosition']).texture,
	});

	Object.defineProperty(base, 'textureVelocity', {
		get:() => rttTextures['velocityTexture'].clone(),
	});

	Object.defineProperty(base, 'randomTexture', {
		get:() => randomTexture,
	});

	Object.defineProperty(base, 'amount', {
		get:() => AMOUNT,
	});

	return base;
};

export default ParticleEngine;

