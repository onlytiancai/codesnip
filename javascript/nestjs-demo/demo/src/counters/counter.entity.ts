import { Entity, Column, PrimaryGeneratedColumn } from 'typeorm';

@Entity("nestjs_counter")
export class Counter {
  @PrimaryGeneratedColumn()
  id: number;

  @Column()
  counter: number;
}