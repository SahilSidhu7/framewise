import type { IAuthenticateGeneric, ICredentialTestRequest, ICredentialType, INodeProperties } from 'n8n-workflow';

export class FramewiseApi implements ICredentialType {
	name = 'framewiseApi';
	displayName = 'Framewise API';
	icon = 'file:framewise.svg' as const;
	properties: INodeProperties[] = [
		{
			displayName: 'Engine URL',
			name: 'baseUrl',
			type: 'string',
			default: 'http://127.0.0.1:8765',
			description: 'Where `framewise serve` is running. From n8n in Docker on the same PC use http://host.docker.internal:8765.',
		},
		{
			displayName: 'API Key',
			name: 'apiKey',
			type: 'string',
			typeOptions: { password: true },
			default: '',
			description: 'Create one with `framewise keys create n8n`',
		},
	];

	authenticate: IAuthenticateGeneric = {
		type: 'generic',
		properties: { headers: { Authorization: '=Bearer {{$credentials.apiKey}}' } },
	};

	test: ICredentialTestRequest = {
		request: { baseURL: '={{$credentials.baseUrl}}', url: '/v1/me' },
	};
}
