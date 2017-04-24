#include <semaphore.h>

#pragma warning( disable : 4100)

int sem_init(sem_t *sem, int pshared, unsigned int value)
{
  sem->sem = CreateSemaphore(NULL, (long)(value), SEM_VALUE_MAX, NULL);
  if (sem->sem != NULL)
    return 0;
  else
    return ENOMEM;
}

#pragma warning( default : 4100)

int sem_destroy(sem_t *sem)
{
  return !CloseHandle(sem->sem);
}

int sem_wait(sem_t *sem)
{
  return WaitForSingleObject(sem->sem, INFINITE);
}

int sem_post(sem_t *sem)
{
  if (!ReleaseSemaphore(sem->sem, 1, NULL))
    return EINVAL;
  else
    return 0;
}
