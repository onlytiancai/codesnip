import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { TypeOrmModule } from '@nestjs/typeorm';
import { DataSource } from 'typeorm';
import { Counter } from './counters/counter.entity';
import { CountersModule} from './counters/counter.module'
import { CounterService } from './counters/counter.service';

@Module({
  imports: [
    TypeOrmModule.forRoot({
      type: 'mysql',
      host: 'localhost',
      port: 3306,
      username: 'root',
      password: 'password',
      database: 'test',
      entities: [Counter],
      synchronize: true,
    }),
    CountersModule,
  ],
  controllers: [AppController],
  providers: [AppService, CounterService],
})
export class AppModule {
  constructor(private dataSource: DataSource) {}
 }
