from sqlalchemy.ext.asyncio import AsyncSession


async def commit(db: AsyncSession) -> None:
    await db.commit()


async def rollback(db: AsyncSession) -> None:
    await db.rollback()


async def refresh(db: AsyncSession, instance: object) -> None:
    await db.refresh(instance)
