# Org And Repo Shape

## Recommendation

Use `PwnKit Labs` as the umbrella.

Recommended structure:

- `PwnKit`
  security product
- `PwnKit Labs`
  research umbrella and advanced agent systems
- `Noeris`
  private repo for the autonomous ML/LLM research engine

## Why Not Put It Directly In The Main Product Repo

- the scope is broader than security
- the architecture is more like R&D infrastructure than product feature code
- the repo needs freedom to evolve faster than the main product
- it may later become its own public project or company surface

## Why Not Put Everything Under A Generic Personal Umbrella

You can technically host it anywhere, but `PwnKit Labs` creates a cleaner story:

- shared founder identity
- shared technical lineage
- room for multiple products
- clear distinction between product and research infrastructure

## Practical Call

For now:

- keep `noeris` private
- treat it as a `PwnKit Labs` incubation repo
- avoid forcing the public company naming decision yet

That is the highest-leverage setup while the system is still proving itself.
