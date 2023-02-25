from apify import Actor


async def main():
    async with Actor() as actor:
        # Get the value of the actor input
        actor_input = await actor.get_input() or {}

        # Structure of input is defined in INPUT_SCHEMA.json
        first_number = actor_input.get('first_number')
        second_number = actor_input.get('second_number')

        # Structure of output is defined in .actor/actor.json
        await actor.push_data([
            {
                'first_number': first_number,
                'second_number': second_number,
                'sum': result,
            },
        ])
