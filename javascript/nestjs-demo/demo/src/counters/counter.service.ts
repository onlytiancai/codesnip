import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository, UpdateResult } from 'typeorm';
import { Counter } from './counter.entity';

@Injectable()
export class CounterService {
  constructor(
    @InjectRepository(Counter)
    private counterRepository: Repository<Counter>,
  ) {}

  findAll(): Promise<Counter[]> {
    return this.counterRepository.find();
  }

  increment(id: number): Promise<UpdateResult> {
    return this.counterRepository.increment({id: id}, 'counter', 1);
  }

  findOne(id: number): Promise<Counter> {
    return this.counterRepository.findOneBy({ id });
  }

  async remove(id: string): Promise<void> {
    await this.counterRepository.delete(id);
  }
}