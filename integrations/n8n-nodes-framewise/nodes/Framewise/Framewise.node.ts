import type {
	IDataObject,
	IExecuteFunctions,
	IHttpRequestOptions,
	INodeExecutionData,
	INodeProperties,
	INodeType,
	INodeTypeDescription,
} from 'n8n-workflow';
import { NodeOperationError } from 'n8n-workflow';

const RECIPES: Record<string, string> = {
	makeReels: 'reel',
	findHighlights: 'highlights',
	addCaptions: 'captions',
	editVideo: 'video_edit',
	editPhoto: 'photo_edit',
	renderProgram: 'render_program',
	transcribe: 'transcribe',
};
const JOB_OPS = Object.keys(RECIPES);
const show = (ops: string[]) => ({ show: { operation: ops } });

const properties: INodeProperties[] = [
	{
		displayName: 'Operation',
		name: 'operation',
		type: 'options',
		noDataExpression: true,
		default: 'makeReels',
		options: [
			{ name: 'Make Reels', value: 'makeReels', action: 'Make reels from a long video', description: 'Best moments as vertical clips with captions' },
			{ name: 'Find Highlights', value: 'findHighlights', action: 'Find highlights in a video' },
			{ name: 'Add Captions', value: 'addCaptions', action: 'Add captions to a video', description: 'Burned-in captions and an .srt file' },
			{ name: 'Edit Video', value: 'editVideo', action: 'Edit a video from a prompt' },
			{ name: 'Edit Photo', value: 'editPhoto', action: 'Edit photos from a prompt' },
			{ name: 'Render Program', value: 'renderProgram', action: 'Render an edit DSL program' },
			{ name: 'Transcribe', value: 'transcribe', action: 'Transcribe speech' },
			{ name: 'Get Job', value: 'getJob', action: 'Get a job' },
			{ name: 'Cancel Job', value: 'cancelJob', action: 'Cancel a job' },
		],
	},
	{
		displayName: 'Input',
		name: 'inputMode',
		type: 'options',
		default: 'binary',
		displayOptions: show(JOB_OPS),
		options: [
			{ name: 'Binary Data From the Previous Node', value: 'binary' },
			{ name: 'URLs or Uploaded File IDs', value: 'refs' },
		],
	},
	{
		displayName: 'Binary Properties',
		name: 'binaryProperties',
		type: 'string',
		default: 'data',
		displayOptions: { show: { operation: JOB_OPS, inputMode: ['binary'] } },
		description: 'Comma-separated binary property names; files are numbered 1..n in this order',
	},
	{
		displayName: 'Inputs',
		name: 'refs',
		type: 'string',
		default: '',
		displayOptions: { show: { operation: JOB_OPS, inputMode: ['refs'] } },
		description: 'Comma-separated http(s) URLs or file IDs (f_...)',
	},
	{
		displayName: 'Prompt',
		name: 'prompt',
		type: 'string',
		typeOptions: { rows: 2 },
		default: '',
		required: true,
		displayOptions: show(['editVideo']),
		placeholder: 'fun upbeat travel recap',
	},
	{
		displayName: 'Prompt',
		name: 'prompt',
		type: 'string',
		default: '',
		displayOptions: show(['editPhoto']),
		placeholder: 'warmer, brighter, ready for Instagram',
	},
	{
		displayName: 'Program',
		name: 'program',
		type: 'string',
		typeOptions: { rows: 6 },
		default: 'aspect 9:16\nclip 1 0-10\ncaptions bold_pop',
		required: true,
		displayOptions: show(['renderProgram']),
		description: 'Video Edit DSL (or Photo Edit DSL for photos)',
	},
	{
		displayName: 'Style',
		name: 'style',
		type: 'options',
		default: '',
		displayOptions: show(['makeReels', 'editVideo', 'editPhoto']),
		options: [
			{ name: 'Auto', value: '' },
			{ name: 'Cinematic', value: 'cinematic' },
			{ name: 'Energetic Reel', value: 'energetic_reel' },
			{ name: 'Vlog', value: 'vlog' },
			{ name: 'Documentary', value: 'documentary' },
			{ name: 'Travel', value: 'travel' },
			{ name: 'Funny', value: 'funny' },
			{ name: 'Gaming', value: 'gaming' },
			{ name: 'Sports Highlights', value: 'sports_highlights' },
			{ name: 'Tutorial', value: 'tutorial' },
			{ name: 'Trailer', value: 'trailer' },
			{ name: 'Music Video', value: 'music_video' },
			{ name: 'Calm / Aesthetic', value: 'calm_aesthetic' },
			{ name: 'Photo: Natural', value: 'natural' },
			{ name: 'Photo: Portrait', value: 'portrait' },
			{ name: 'Photo: Food', value: 'food' },
			{ name: 'Photo: Product', value: 'product' },
			{ name: 'Photo: Moody', value: 'moody' },
			{ name: 'Photo: Vintage', value: 'vintage' },
			{ name: 'Photo: Black & White', value: 'black_white' },
		],
	},
	{
		displayName: 'For',
		name: 'platform',
		type: 'options',
		default: '',
		displayOptions: show(['makeReels', 'editVideo', 'editPhoto']),
		options: [
			{ name: 'Auto', value: '' },
			{ name: 'TikTok', value: 'tiktok' },
			{ name: 'Instagram Reel', value: 'instagram_reel' },
			{ name: 'YouTube Short', value: 'youtube_short' },
			{ name: 'Story', value: 'story' },
			{ name: 'YouTube', value: 'youtube' },
			{ name: 'Square Post', value: 'square_post' },
			{ name: 'Portrait Post (4:5)', value: 'portrait_post' },
			{ name: 'Movie', value: 'movie' },
			{ name: 'Photo: Instagram Post', value: 'instagram_post' },
			{ name: 'Photo: Wallpaper', value: 'wallpaper' },
			{ name: 'Photo: Print', value: 'print' },
		],
	},
	{
		displayName: 'Wait for Result',
		name: 'wait',
		type: 'boolean',
		default: true,
		displayOptions: show(JOB_OPS),
		description: 'Whether to wait until the job is finished. Turn off when you use a webhook instead.',
	},
	{
		displayName: 'Download Results',
		name: 'download',
		type: 'boolean',
		default: true,
		displayOptions: show([...JOB_OPS, 'getJob']),
		description: 'Whether to attach the result files as binary data (file_1, file_2, ...)',
	},
	{
		displayName: 'Job ID',
		name: 'jobId',
		type: 'string',
		default: '',
		required: true,
		displayOptions: show(['getJob', 'cancelJob']),
	},
	{
		displayName: 'Options',
		name: 'extra',
		type: 'collection',
		placeholder: 'Add option',
		default: {},
		displayOptions: show(JOB_OPS),
		options: [
			{ displayName: 'Count', name: 'count', type: 'number', default: 3, description: 'Reels or highlights to make' },
			{ displayName: 'Length (Seconds)', name: 'length', type: 'number', default: 30 },
			{ displayName: 'Versions', name: 'versions', type: 'number', default: 1, typeOptions: { minValue: 1, maxValue: 5 } },
			{ displayName: 'Captions', name: 'captions', type: 'boolean', default: true },
			{ displayName: 'Music', name: 'music', type: 'boolean', default: true },
			{
				displayName: 'Caption Style', name: 'captionStyle', type: 'options', default: 'bold_pop',
				options: ['clean', 'bold_pop', 'cinematic', 'boxed', 'title', 'minimal'].map((v) => ({ name: v, value: v })),
			},
			{
				displayName: 'Quality', name: 'quality', type: 'options', default: 'standard',
				options: [{ name: 'Draft (Fast)', value: 'draft' }, { name: 'Standard', value: 'standard' }, { name: 'High', value: 'high' }],
			},
			{ displayName: 'Webhook URL', name: 'webhook', type: 'string', default: '', description: 'Called when the job finishes (signed, see docs)' },
			{ displayName: 'Timeout (Minutes)', name: 'timeout', type: 'number', default: 60 },
			{ displayName: 'Recipe Options (JSON)', name: 'optionsJson', type: 'json', default: '{}', description: 'Any other recipe option, see GET /v1/recipes' },
		],
	},
];

export class Framewise implements INodeType {
	description: INodeTypeDescription = {
		displayName: 'Framewise',
		name: 'framewise',
		icon: 'file:framewise.svg',
		group: ['transform'],
		version: 1,
		subtitle: '={{$parameter["operation"]}}',
		description: 'Make reels, captions and AI edits with the Framewise engine on your own computer',
		defaults: { name: 'Framewise' },
		inputs: ['main'],
		outputs: ['main'],
		credentials: [{ name: 'framewiseApi', required: true }],
		properties,
	};

	async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
		const items = this.getInputData();
		const out: INodeExecutionData[] = [];
		const creds = await this.getCredentials('framewiseApi');
		const base = String(creds.baseUrl).replace(/\/+$/, '');
		const call = async (opts: Partial<IHttpRequestOptions> & { url: string }) =>
			this.helpers.httpRequestWithAuthentication.call(this, 'framewiseApi', {
				json: true,
				...opts,
				url: base + opts.url,
			} as IHttpRequestOptions);

		for (let i = 0; i < items.length; i++) {
			try {
				const op = this.getNodeParameter('operation', i) as string;
				let job: IDataObject;
				if (op === 'getJob' || op === 'cancelJob') {
					const id = this.getNodeParameter('jobId', i) as string;
					job = op === 'getJob'
						? await call({ method: 'GET', url: `/v1/jobs/${encodeURIComponent(id)}` })
						: await call({ method: 'POST', url: `/v1/jobs/${encodeURIComponent(id)}/cancel` });
				} else {
					const extra = this.getNodeParameter('extra', i, {}) as IDataObject;
					const inputs: string[] = [];
					if (this.getNodeParameter('inputMode', i) === 'binary') {
						const props = String(this.getNodeParameter('binaryProperties', i)).split(',').map((s) => s.trim()).filter(Boolean);
						for (const prop of props) {
							const meta = this.helpers.assertBinaryData(i, prop);
							const data = await this.helpers.getBinaryDataBuffer(i, prop);
							const up = await call({
								method: 'PUT', url: '/v1/files', qs: { name: meta.fileName || `${prop}.bin` }, body: data,
								headers: { 'Content-Type': 'application/octet-stream' }, json: false,
							});
							inputs.push((typeof up === 'string' ? JSON.parse(up) : up).file_id);
						}
					} else {
						inputs.push(...String(this.getNodeParameter('refs', i)).split(',').map((s) => s.trim()).filter(Boolean));
					}
					if (!inputs.length) throw new NodeOperationError(this.getNode(), 'No input files', { itemIndex: i });

					const options: IDataObject = typeof extra.optionsJson === 'string' ? JSON.parse(extra.optionsJson || '{}') : (extra.optionsJson as IDataObject) || {};
					if (op === 'makeReels' || op === 'findHighlights') {
						if (extra.count !== undefined) options.count = extra.count;
						if (extra.length !== undefined) options.length = extra.length;
					}
					if (op === 'addCaptions' && extra.captionStyle) options.style = extra.captionStyle;
					if (op === 'renderProgram') options.program = this.getNodeParameter('program', i);
					const output: IDataObject = {};
					for (const k of ['versions', 'captions', 'music', 'quality']) if (extra[k] !== undefined) output[k] = extra[k];
					if (op === 'editVideo' && extra.length !== undefined) output.length = extra.length;
					const body: IDataObject = { recipe: RECIPES[op], inputs, options };
					if (['makeReels', 'editVideo', 'editPhoto'].includes(op)) {
						const style = this.getNodeParameter('style', i, '') as string;
						const platform = this.getNodeParameter('platform', i, '') as string;
						if (style) body.style = style;
						if (platform) output.platform = platform;
					}
					if (op === 'editVideo' || op === 'editPhoto') {
						const prompt = this.getNodeParameter('prompt', i, '') as string;
						if (prompt) body.prompt = prompt;
					}
					if (Object.keys(output).length) body.output = output;
					if (extra.webhook) body.webhook = extra.webhook;
					job = await call({ method: 'POST', url: '/v1/jobs', body });

					if (this.getNodeParameter('wait', i)) {
						const until = Date.now() + Number(extra.timeout ?? 60) * 60_000;
						while (!['done', 'failed', 'cancelled'].includes(String(job.state)) && Date.now() < until) {
							job = await call({ method: 'GET', url: `/v1/jobs/${job.id}/wait`, qs: { timeout: 60 } });
						}
						if (job.state === 'failed') throw new NodeOperationError(this.getNode(), `Framewise job failed: ${job.error}`, { itemIndex: i });
					}
				}

				const item: INodeExecutionData = { json: job, pairedItem: { item: i } };
				if (job.state === 'done' && op !== 'cancelJob' && this.getNodeParameter('download', i, true)) {
					item.binary = {};
					const outputs = (job.outputs as IDataObject[]) || [];
					for (let k = 0; k < outputs.length; k++) {
						const o = outputs[k];
						const data = await call({ method: 'GET', url: String(o.url), json: false, encoding: 'arraybuffer' });
						item.binary[`file_${k + 1}`] = await this.helpers.prepareBinaryData(Buffer.from(data as ArrayBuffer), String(o.name));
					}
				}
				out.push(item);
			} catch (error) {
				if (this.continueOnFail()) {
					out.push({ json: { error: (error as Error).message }, pairedItem: { item: i } });
					continue;
				}
				throw error;
			}
		}
		return [out];
	}
}
