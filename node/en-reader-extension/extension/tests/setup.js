global.chrome = {
  runtime: {
    sendMessage: jest.fn(),
    onMessage: {
      addListener: jest.fn(),
    },
  },
};

global.Audio = class Audio {
  constructor() {
    this.play = jest.fn();
  }
};
