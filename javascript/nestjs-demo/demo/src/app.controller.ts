import { Controller, Get } from '@nestjs/common';
import { AppService } from './app.service';
import { CounterService } from './counters/counter.service';

@Controller()
export class AppController {
  constructor(
    private readonly appService: AppService,
    private readonly counterService: CounterService,
    ) {}

  @Get()
  async getHello(): Promise<string> {
    let ret = this.appService.getHello()
    await this.counterService.increment(1)
    const counter = await this.counterService.findOne(1)
    ret = ret.replace('{counter}', counter.counter.toString())
    return ret
  }
}
