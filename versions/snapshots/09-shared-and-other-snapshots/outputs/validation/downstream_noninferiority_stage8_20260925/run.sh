#!/bin/bash
cd /home/user/Robotic/project2
O=outputs/validation/downstream_noninferiority_stage8_20260925
S=outputs/validation/downstream_noninferiority_20260924
exec 9>/tmp/project2_gpu.lock; flock 9
for pair in "open door(door)|open_door_door" "pick up from(toy_box,floors)|pick_up_from_toy_box_floors" "place in(toy_box,car)|place_in_toy_box_car" "close lid(car)|close_lid_car"; do
  key="${pair%%|*}"; tag="${pair##*|}"
  .conda-envs/lilawam/bin/python scripts/check_openlid_replay.py --checkpoint outputs/training/stage8_motionw_manip_20260925/checkpoint_final.pt \
    --key "$key" --out $O/stage8_$tag --samples 8 --device cuda --skip-missing --admit-against $S/stage3_$tag > $O/stage8_$tag.log 2>&1
  echo "stage8 $key rc=$?" >> $O/run.log
done
