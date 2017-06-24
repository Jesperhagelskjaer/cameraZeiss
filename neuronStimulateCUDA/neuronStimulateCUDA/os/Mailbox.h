#pragma once
#include "Semaphore.h"
#include<queue>

template<class Item>
class Mailbox
{
public:
  Mailbox(const LONG cap);
  ~Mailbox();
  void put(Item n);
  Item get();

private:
  LONG capacity;
  Semaphore* getSemaphore;
  Semaphore* putSemaphore;
  std::queue<Item>* queue;
};

#include "Mailbox.tmpl"