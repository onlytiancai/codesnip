import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { CounterService } from './counter.service';
import { Counter } from './counter.entity';

@Module({
  imports: [TypeOrmModule.forFeature([Counter])],
  providers: [CounterService],
  exports: [TypeOrmModule]
})
export class CountersModule {}