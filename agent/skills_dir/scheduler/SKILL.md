name: scheduler
description: Skill for analyzing the user's daily schedule via knowledge base logged events and event retrieval, identifying high-risk glucose moments, and adding glucose measurement reminders to the calendar upon user request.
instructions: |
  When initiated by the user for schedule review and reminder planning, follow this workflow:
  1. **Schedule Retrieval**: Retrieve planned calendar events from the vector knowledge base using `get_calendar_events_from_kb` tool. Retrieve all event groups, unless user specified otherwise.
  2. **High-Risk Identification**: Evaluate which events (such as exercise sessions, high-stress meetings, or irregular meal times) could significantly impact glucose levels and overall well-being. Consider such events as high-risk
  3. **Reminder Formulation**: Generate a list of recommended reminders positioned around these high-risk moments and add them to the calendar using `add_notification_events_to_kb` tool. The reminder should be created, if during the time of an event there is a high-risk of glucose level abnormalities.
  Such reminders should notify user to keep track of their glucose levels at the specified times. Below are the guidelines for `add_notification_events_to_kb` tool:  
  - **name**: By default notifications are called: `Check glucose level`. User can override this in their query, in this case - use user-defined name.
  - **start_time**: Time of the notification. The time should be decided based on timings of evaluated high-risk events. The notification could be placed right before, right after, or during the high-risk event.
  - **end_time**: Optional. Defines the limit for the range of time, when notification is relevant, for example if user should keep track of their glucose levels during this whole time. Decided based on the corresponding high-risk event. If ignored - defaults to start_time.
  - **description**: Optional. May include brief reasoning for why the notification was created and what particular danger does the corresponding high-risk event posses.

tools: [add_notification_events_to_kb]