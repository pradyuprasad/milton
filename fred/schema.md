
## Tables

### `categories`
- `id` INTEGER PRIMARY KEY AUTOINCREMENT
- `fred_id` INTEGER UNIQUE
- `name` TEXT
- `parent_id` INTEGER (Foreign Key to `categories.id`)

### `releases`
- `id` INTEGER PRIMARY KEY AUTOINCREMENT
- `fred_id` INTEGER UNIQUE
- `realtime_start` TEXT
- `realtime_end` TEXT
- `name` TEXT
- `press_release` BOOLEAN
- `link` TEXT

### `series`
- `id` INTEGER PRIMARY KEY AUTOINCREMENT
- `fred_id` TEXT UNIQUE
- `realtime_start` TEXT
- `realtime_end` TEXT
- `title` TEXT
- `observation_start` TEXT
- `observation_end` TEXT
- `frequency` TEXT
- `frequency_short` TEXT
- `units` TEXT
- `units_short` TEXT
- `seasonal_adjustment` TEXT
- `seasonal_adjustment_short` TEXT
- `last_updated` TEXT
- `popularity` INTEGER
- `notes` TEXT

### `tags`
- `id` INTEGER PRIMARY KEY AUTOINCREMENT
- `fred_id` INTEGER UNIQUE
- `name` TEXT
- `group_id` TEXT
- `notes` TEXT
- `created` TEXT
- `popularity` INTEGER
- `series_count` INTEGER

### `sources`
- `id` INTEGER PRIMARY KEY AUTOINCREMENT
- `fred_id` INTEGER UNIQUE
- `realtime_start` TEXT
- `realtime_end` TEXT
- `name` TEXT
- `link` TEXT

## Relationship Tables
These tables establish many-to-many relationships between the primary tables.

### `category_series`
- `category_id` INTEGER (Foreign Key to `categories.id`)
- `series_id` INTEGER (Foreign Key to `series.id`)
- PRIMARY KEY (`category_id`, `series_id`)

### `release_series`
- `release_id` INTEGER (Foreign Key to `releases.id`)
- `series_id` INTEGER (Foreign Key to `series.id`)
- PRIMARY KEY (`release_id`, `series_id`)

### `series_tags`
- `series_id` INTEGER (Foreign Key to `series.id`)
- `tag_id` INTEGER (Foreign Key to `tags.id`)
- PRIMARY KEY (`series_id`, `tag_id`)

### `category_tags`
- `category_id` INTEGER (Foreign Key to `categories.id`)
- `tag_id` INTEGER (Foreign Key to `tags.id`)
- PRIMARY KEY (`category_id`, `tag_id`)

### `release_tags`
- `release_id` INTEGER (Foreign Key to `releases.id`)
- `tag_id` INTEGER (Foreign Key to `tags.id`)
- PRIMARY KEY (`release_id`, `tag_id`)

### `release_sources`
- `release_id` INTEGER (Foreign Key to `releases.id`)
- `source_id` INTEGER (Foreign Key to `sources.id`)
- PRIMARY KEY (`release_id`, `source_id`)
