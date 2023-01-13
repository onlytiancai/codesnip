const parse = require('./sum');

test('empty', () => {
    expect(parse('')).toEqual({
        tags: [],
        data: []
    });
});

test('default', () => {
    expect(parse(`
    111
    222
    `)).toEqual({
        tags: [],
        data: [
            {
                tags: [],
                text: '111',
            },
            {
                tags: [],
                text: '222'
            }
        ]
    });
});

test('tags', () => {
    expect(parse(`
    [a]111
    [b][c]222
    `)).toEqual({
        tags: ['a', 'b', 'c'],
        data: [
            {
                tags: ['a'],
                text: '111',
            },
            {
                tags: ['b','c'],
                text: '222'
            }
        ]
    });
});