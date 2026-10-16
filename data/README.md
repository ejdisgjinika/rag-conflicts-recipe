# Recipe data

The dataset contains 1,165 rows and 200 recipe names. Each recipe variant is
identified by the pair `(RecipeName, Code)`.

| File | Required columns |
| --- | --- |
| `dataset_recipes.csv` | `RecipeName`, `Code`, `Text` |
| `cleaned_ingredients.csv` | `Ingredients` |

The files have matching row order: each ingredient entry corresponds to the recipe
variant at the same position. The loader validates required fields, row counts and
unique variant identifiers, and filters invalid ingredient payloads.

The controlled experiment selects 100 eligible six-variant recipes alphabetically
and forms groups with six, five, four and three variants, producing 400 cases.

The accompanying paper cites
[Food.com – Recipes and Reviews](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews)
by Kaggle user `irkaal` as its data source. See [data licensing](../DATA_LICENSE.md).
