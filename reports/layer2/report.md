# Layer 2 evaluation

```
{
 "counts": {
  "receive": 808,
  "release": 808,
  "touch": 347,
  "unattributed": 174,
  "lost": 151,
  "found": 150,
  "artifact": 191,
  "stop": 195,
  "kick": 69
 },
 "precision": {
  "n_free": 521,
  "attributed_share": 0.6660268714011516,
  "near_1_0": 0.2571976967370441,
  "random_near_1_0": 0.226,
  "near_1_5": 0.4049904030710173,
  "random_near_1_5": 0.3728,
  "near_2_5": 0.6660268714011516,
  "random_near_2_5": 0.599
 },
 "completeness": {
  "free_stretches": 1373,
  "unexplained_speedup": 0.04515659140568099,
  "unexplained_turn": 0.1099781500364166,
  "unexplained_any": 0.1551347414420976
 },
 "plausibility": {
  "ball_known_minutes": 62.712666666666664,
  "receives_per_min": 12.884159499941532,
  "free_touches_per_min": 5.533172458514495,
  "next_toucher_same_team": 0.7349514563106796,
  "next_receiver_same_team": 0.7846385542168675,
  "pass_length_m_p25_p50_p75": [
   4.622951446956726,
   13.955285926749687,
   20.539445409827554
  ],
  "held_s_p25_p50_p75": [
   0.52,
   0.9,
   1.6
  ],
  "share_of_known_ball_with_a_player": 0.28746983597146775
 },
 "sweep": [
  {
   "param": "CONTROL_D",
   "value": 1.0,
   "unexplained_any": 0.1876675603217158,
   "near_1_5": 0.5550786838340487,
   "random_1_5": 0.4302,
   "attributed": 0.7510729613733905,
   "next_receiver_same_team": 0.8175182481751825,
   "receives_per_min": 11.241748078538095,
   "free_touches_per_min": 8.371514526570921
  },
  {
   "param": "CONTROL_D",
   "value": 1.25,
   "unexplained_any": 0.16666666666666666,
   "near_1_5": 0.4728813559322034,
   "random_1_5": 0.3926,
   "attributed": 0.7050847457627119,
   "next_receiver_same_team": 0.7929373996789727,
   "receives_per_min": 12.246329821726606,
   "free_touches_per_min": 6.633428653435245
  },
  {
   "param": "CONTROL_D",
   "value": 1.5,
   "unexplained_any": 0.1551347414420976,
   "near_1_5": 0.4049904030710173,
   "random_1_5": 0.3728,
   "attributed": 0.6660268714011516,
   "next_receiver_same_team": 0.7846385542168675,
   "receives_per_min": 12.884159499941532,
   "free_touches_per_min": 5.533172458514495
  },
  {
   "param": "CONTROL_D",
   "value": 2.0,
   "unexplained_any": 0.15186028853454822,
   "near_1_5": 0.3905579399141631,
   "random_1_5": 0.3634,
   "attributed": 0.628755364806867,
   "next_receiver_same_team": 0.7586705202312138,
   "receives_per_min": 13.346586016647356,
   "free_touches_per_min": 4.672102392924343
  },
  {
   "param": "CONTROL_RELV",
   "value": 2.0,
   "unexplained_any": 0.20141342756183744,
   "near_1_5": 0.514866979655712,
   "random_1_5": 0.4228,
   "attributed": 0.7276995305164319,
   "next_receiver_same_team": 0.8214971209213052,
   "receives_per_min": 10.699592852055407,
   "free_touches_per_min": 7.414770009248531
  },
  {
   "param": "CONTROL_RELV",
   "value": 3.0,
   "unexplained_any": 0.1551347414420976,
   "near_1_5": 0.4049904030710173,
   "random_1_5": 0.3728,
   "attributed": 0.6660268714011516,
   "next_receiver_same_team": 0.7846385542168675,
   "receives_per_min": 12.884159499941532,
   "free_touches_per_min": 5.533172458514495
  },
  {
   "param": "CONTROL_RELV",
   "value": 4.0,
   "unexplained_any": 0.14074074074074075,
   "near_1_5": 0.32229580573951433,
   "random_1_5": 0.3404,
   "attributed": 0.6158940397350994,
   "next_receiver_same_team": 0.758893280632411,
   "receives_per_min": 14.287384792014374,
   "free_touches_per_min": 4.448862005549119
  },
  {
   "param": "TOUCH_SPEED_MIN_PROMINENCE",
   "value": 1.0,
   "unexplained_any": 0.15010570824524314,
   "near_1_5": 0.3756177924217463,
   "random_1_5": 0.3728,
   "attributed": 0.627677100494234,
   "next_receiver_same_team": 0.7846385542168675,
   "receives_per_min": 12.884159499941532,
   "free_touches_per_min": 6.075327684997183
  },
  {
   "param": "TOUCH_SPEED_MIN_PROMINENCE",
   "value": 2.0,
   "unexplained_any": 0.1551347414420976,
   "near_1_5": 0.4049904030710173,
   "random_1_5": 0.3728,
   "attributed": 0.6660268714011516,
   "next_receiver_same_team": 0.7846385542168675,
   "receives_per_min": 12.884159499941532,
   "free_touches_per_min": 5.533172458514495
  },
  {
   "param": "TOUCH_SPEED_MIN_PROMINENCE",
   "value": 3.0,
   "unexplained_any": 0.16299559471365638,
   "near_1_5": 0.41030927835051545,
   "random_1_5": 0.3728,
   "attributed": 0.6762886597938145,
   "next_receiver_same_team": 0.7846385542168675,
   "receives_per_min": 12.884159499941532,
   "free_touches_per_min": 5.230203361362404
  },
  {
   "param": "TOUCH_TURN_DEG",
   "value": 20.0,
   "unexplained_any": 0.127208480565371,
   "near_1_5": 0.41652892561983473,
   "random_1_5": 0.3728,
   "attributed": 0.6760330578512397,
   "next_receiver_same_team": 0.7846385542168675,
   "receives_per_min": 12.884159499941532,
   "free_touches_per_min": 6.521808459747632
  },
  {
   "param": "TOUCH_TURN_DEG",
   "value": 30.0,
   "unexplained_any": 0.1551347414420976,
   "near_1_5": 0.4049904030710173,
   "random_1_5": 0.3728,
   "attributed": 0.6660268714011516,
   "next_receiver_same_team": 0.7846385542168675,
   "receives_per_min": 12.884159499941532,
   "free_touches_per_min": 5.533172458514495
  },
  {
   "param": "TOUCH_TURN_DEG",
   "value": 45.0,
   "unexplained_any": 0.20121951219512196,
   "near_1_5": 0.4043062200956938,
   "random_1_5": 0.3728,
   "attributed": 0.6770334928229665,
   "next_receiver_same_team": 0.7846385542168675,
   "receives_per_min": 12.884159499941532,
   "free_touches_per_min": 4.5126449733706115
  },
  {
   "param": "TOUCH_TURN_MAX_DEG",
   "value": 60.0,
   "unexplained_any": 0.1551347414420976,
   "near_1_5": 0.4117647058823529,
   "random_1_5": 0.3728,
   "attributed": 0.6719457013574661,
   "next_receiver_same_team": 0.7846385542168675,
   "receives_per_min": 12.884159499941532,
   "free_touches_per_min": 4.7358853607458355
  },
  {
   "param": "TOUCH_TURN_MAX_DEG",
   "value": 90.0,
   "unexplained_any": 0.1551347414420976,
   "near_1_5": 0.4049904030710173,
   "random_1_5": 0.3728,
   "attributed": 0.6660268714011516,
   "next_receiver_same_team": 0.7846385542168675,
   "receives_per_min": 12.884159499941532,
   "free_touches_per_min": 5.533172458514495
  },
  {
   "param": "TOUCH_TURN_MAX_DEG",
   "value": 120.0,
   "unexplained_any": 0.1551347414420976,
   "near_1_5": 0.39896373056994816,
   "random_1_5": 0.3728,
   "attributed": 0.6666666666666666,
   "next_receiver_same_team": 0.7846385542168675,
   "receives_per_min": 12.884159499941532,
   "free_touches_per_min": 6.155056394774049
  }
 ]
}
```
