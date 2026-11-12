from eltdx import TdxClient

client = TdxClient(timeout=3)
profile = client.f10.company_profile("000034")
topics = client.f10.hot_topics("000034")
notices = client.f10.announcements("000034")

print(profile.rows[0])
print(topics.rows[:3])
print(notices.rows[:3])
