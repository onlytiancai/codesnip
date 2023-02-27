database:

    CREATE TABLE `feeds` (
      `id` int(11) NOT NULL AUTO_INCREMENT,
      `feed_id` bigint(20) NOT NULL DEFAULT '0',
      `author` varchar(32) NOT NULL DEFAULT '',
      `title` varchar(512) NOT NULL DEFAULT '',
      `summary` varchar(1024) NOT NULL DEFAULT '',
      `published` datetime NOT NULL DEFAULT '1970-01-01 00:00:00',
      PRIMARY KEY (`id`) /*T![clustered_index] CLUSTERED */,
      KEY `ix_feeds_published` (`published`),
      UNIQUE KEY `uq_feeds_feed_id` (`feed_id`)
    )

crontab

    0 * * * * cd /home/ubuntu/src/codesnip/python/hn_feeds && /usr/bin/python3 cron.py >/dev/null 2>&1 &
