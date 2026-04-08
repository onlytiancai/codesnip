# Getting Started with API

This guide will help you understand how to use our API.

## Installation

First, install the SDK:

```bash
npm install our-sdk
```

## Basic Usage

Here is a simple example:

```javascript
const sdk = require('our-sdk');

async function main() {
  const client = new sdk.Client({
    apiKey: 'your-api-key'
  });

  const result = await client.query('Hello world');
  console.log(result);
}
```

## Features

- Fast and reliable
- Easy to use
- Well documented

## Configuration

You can configure the SDK with these options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| timeout | number | 5000 | Request timeout |
| retries | number | 3 | Number of retries |

## Tips

> Always keep your API key secure and never commit it to version control.

## Conclusion

That's it! You're ready to start using the API.
