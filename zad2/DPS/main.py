"""
==========================================
Fuzzy Control Systems: Dynamic Positioning System
==========================================
Authors: Michał Czerwiak s21356, Bartosz Kamiński s20500
To run program you need to install skfuzzy, matplotlib and numpy packages

The system calculates the values for engines of a ship, which is trying to
stay on position on water with a specified drift (speed and angle is given as a parameters)

* Antecednets (Inputs)
   - `driftAngle`
      * from what direction the drift is going to our ship?
   - `driftSpeed`
      * what is the speed of the drift?
* Consequents (Outputs)
   - `verticalThrusters`
      * Universe: What should be the setting for the main engines
      * Fuzzy set: lowAhead, mediumAhead, fullAhead, lowAstern, mediumAstern, fullAstern
   - `horizontalThrusters`
      * Universe: What should be the setting for the horizontal thrusters
      * Fuzzy set: mediumStarboard, fullStarboard, mediumPort, fullPort
* Usage
   - If we have drift with specified angle and speed, the system will compute it and tell us
     how should our ships thrusters work to compensate the drift and stay at the position.
"""
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

driftAngle = ctrl.Antecedent(np.arange(0, 361, 1), 'driftAngle')
driftSpeed = ctrl.Antecedent(np.arange(0, 16, 1), 'driftSpeed')

horizontalThrusters = ctrl.Consequent(np.arange(-10, 11, 1), 'horizontalThrusters')
verticalThrusters = ctrl.Consequent(np.arange(-15, 16, 1), 'verticalThrusters')

"""
For the simulation the driftAngle was split into 8 pieces, 45 degrees each + 4 main directions
(North, East, South, West).
You can imagine this as a compass rose:
"""

driftAngle['north'] = fuzz.trimf(driftAngle.universe, [0, 0, 0])
driftAngle['northToNortheast'] = fuzz.trimf(driftAngle.universe, [0, 45, 45])
driftAngle['northeastToEast'] = fuzz.trimf(driftAngle.universe, [45, 90, 90])
driftAngle['east'] = fuzz.trimf(driftAngle.universe, [90, 90, 90])
driftAngle['eastToSoutheast'] = fuzz.trimf(driftAngle.universe, [90, 135, 135])
driftAngle['southeastToSouth'] = fuzz.trimf(driftAngle.universe, [135, 180, 180])
driftAngle['south'] = fuzz.trimf(driftAngle.universe, [180, 180, 180])
driftAngle['southToSouthwest'] = fuzz.trimf(driftAngle.universe, [180, 225, 225])
driftAngle['southwestToWest'] = fuzz.trimf(driftAngle.universe, [225, 270, 270])
driftAngle['west'] = fuzz.trimf(driftAngle.universe, [270, 270, 270])
driftAngle['westToNorthwest'] = fuzz.trimf(driftAngle.universe, [270, 315, 315])
driftAngle['northwestToNorth'] = fuzz.trimf(driftAngle.universe, [315, 360, 360])

driftSpeed['low'] = fuzz.trimf(driftSpeed.universe, [0, 5, 5])
driftSpeed['medium'] = fuzz.trimf(driftSpeed.universe, [5, 10, 10])
driftSpeed['high'] = fuzz.trimf(driftSpeed.universe, [10, 15, 15])
driftSpeed['nodrift'] = fuzz.trimf(driftSpeed.universe, [0, 0, 0])
"""
Speed of the drift varies between 0 and 15 knots, so it was split into 3 + no drift using .trimf
"""

verticalThrusters['stop'] = fuzz.trimf(verticalThrusters.universe, [0, 0, 0])
verticalThrusters['lowAhead'] = fuzz.trimf(verticalThrusters.universe, [0, 5, 5])
verticalThrusters['mediumAhead'] = fuzz.trimf(verticalThrusters.universe, [5, 10, 10])
verticalThrusters['fullAhead'] = fuzz.trimf(verticalThrusters.universe, [10, 15, 15])
verticalThrusters['lowAstern'] = fuzz.trimf(verticalThrusters.universe, [-5, -5, 0])
verticalThrusters['mediumAstern'] = fuzz.trimf(verticalThrusters.universe, [-10, -10, -5])
verticalThrusters['fullAstern'] = fuzz.trimf(verticalThrusters.universe, [-15, -15, -10])
"""
Verical thrusters settings varies between -15 and 15. It was split into 6 + stop using .trimf
"""
horizontalThrusters['stop'] = fuzz.trimf(horizontalThrusters.universe, [0, 0, 0])
horizontalThrusters['mediumStarboard'] = fuzz.trimf(horizontalThrusters.universe, [0, 5, 5])
horizontalThrusters['fullStarboard'] = fuzz.trimf(horizontalThrusters.universe, [5, 10, 10])
horizontalThrusters['mediumPort'] = fuzz.trimf(horizontalThrusters.universe, [-5, -5, 0])
horizontalThrusters['fullPort'] = fuzz.trimf(horizontalThrusters.universe, [-10, -10, -5])
"""
Horizontal thrusters settings varies between -10 and 10. It was split into 4 + stop setting using .trimf
"""

driftAngle.view()
"""
Plots membership function for the driftAngle
"""

driftSpeed.view()
"""
Plots membership function for the driftSpeed
"""

horizontalThrusters.view()
"""
Plots membership function for the horizontalThrusters
"""
verticalThrusters.view()
"""
Plots membership function for the verticalThrusters

Fuzzy rules
-----------

1. If the drift direction is from North with low speed, then horizontal thrusters should be stopped and vertical 
   thrusters should work low ahead.
2. If the drift direction is from North with medium speed, then horizontal thrusters should be stopped and vertical 
   thrusters should work medium ahead.
3. If the drift direction is from North with high speed, then horizontal thrusters should be stopped and vertical 
   thrusters should work full ahead.
4. If the drift direction is from North to NorthEast with low speed, then horizontal thrusters should work medium 
   starboard and vertical thrusters low ahead.
5. If the drift direction is from North to NorthEast with medium speed, then horizontal thrusters should work medium 
   starboard and vertical thrusters medium ahead.
6. If the drift direction is from North to NorthEast with high speed, then horizontal thrusters should work medium 
   starboard and vertical thrusters full ahead.
7. If the drift direction is from NorthEast to East with low speed, then horizontal thrusters should work medium 
   starboard and vertical thrusters low ahead.
8. If the drift direction is from NorthEast to East with medium speed, then horizontal thrusters should work medium 
   starboard and vertical thrusters low ahead.
9. If the drift direction is from NorthEast to East with high speed, then horizontal thrusters should work full 
   starboard and vertical thrusters low ahead.
10. If the drift direction is from East with low speed, then horizontal thrusters should work medium starboard 
    and vertical thrusters should be stopped.
11. If the drift direction is from East with medium speed, then horizontal thrusters should work medium starboard 
    and vertical thrusters should be stopped.
12. If the drift direction is from East with high speed, then horizontal thrusters should work full starboard 
    and vertical thrusters should be stopped.
13. If the drift direction is from East to SouthEast with low speed, then horizontal thrusters should work medium 
   starboard and vertical thrusters low astern.
14. If the drift direction is from East to SouthEast with medium speed, then horizontal thrusters should work medium 
   starboard and vertical thrusters low astern.
15. If the drift direction is from East to SouthEast with high speed, then horizontal thrusters should work full 
   starboard and vertical thrusters low astern.
16. If the drift direction is from SouthEast to South with low speed, then horizontal thrusters should work medium 
    starboard and vertical thrusters low astern.
17. If the drift direction is from SouthEast to South with medium speed, then horizontal thrusters should work medium 
    starboard and vertical thrusters medium astern.
18. If the drift direction is from SouthEast to South with high speed, then horizontal thrusters should work medium 
    starboard and vertical thrusters full astern.
19. If the drift direction is from South with low speed, then horizontal thrusters should be stopped and vertical 
    thrusters should work low astern.
20. If the drift direction is from South with medium speed, then horizontal thrusters should be stopped and vertical 
    thrusters should work medium astern.
21. If the drift direction is from South with high speed, then horizontal thrusters should be stopped and vertical 
    thrusters should work high astern.
22. If the drift direction is from South to SouthWest with low speed, then horizontal thrusters should work medium 
    port and vertical thrusters low astern.
23. If the drift direction is from South to SouthWest with medium speed, then horizontal thrusters should work medium 
    port and vertical thrusters medium astern.
24. If the drift direction is from South to SouthWest with high speed, then horizontal thrusters should work medium 
    port and vertical thrusters full astern.
25. If the drift direction is from SouthWest to West with low speed, then horizontal thrusters should work medium 
    port and vertical thrusters low astern.
26. If the drift direction is from SouthWest to West with medium speed, then horizontal thrusters should work medium 
    port and vertical thrusters low astern.
27. If the drift direction is from SouthWest to West with high speed, then horizontal thrusters should work full 
    port and vertical thrusters low astern.
28. If the drift direction is from West with low speed, then horizontal thrusters should work medium 
    port and vertical thrusters should be stopped.
29. If the drift direction is from West with low speed, then horizontal thrusters should work medium 
    port and vertical thrusters should be stopped.
30. If the drift direction is from West with low speed, then horizontal thrusters should work full 
    port and vertical thrusters should be stopped.
31. If the drift direction is from West to NorthWest with low speed, then horizontal thrusters should work medium 
    port and vertical thrusters low ahead.
32. If the drift direction is from West to NorthWest with medium speed, then horizontal thrusters should work medium 
    port and vertical thrusters low ahead.
33. If the drift direction is from West to NorthWest with high speed, then horizontal thrusters should work full 
    port and vertical thrusters low ahead.
34. If the drift direction is from NorthWest to North with low speed, then horizontal thrusters should work medium 
    port and vertical thrusters low ahead.
35. If the drift direction is from NorthWest to North with medium speed, then horizontal thrusters should work medium 
    port and vertical thrusters medium ahead.
36. If the drift direction is from NorthWest to North with high speed, then horizontal thrusters should work medium 
    port and vertical thrusters full ahead.
37. If there is no drift, then vertical and horizontal thrusters should be stopped.

Above rules should be given by an expert, who is familiar with navigation issues. The rules are fuzzy and
imprecise. Fuzzy logic counts the optimal parameters for thrusters to compensate the drift.
"""

rule1 = ctrl.Rule(driftAngle['north'] & driftSpeed['low'], (horizontalThrusters['stop'], verticalThrusters['lowAhead']))
rule2 = ctrl.Rule(driftAngle['north'] & driftSpeed['medium'], (horizontalThrusters['stop'], verticalThrusters['mediumAhead']))
rule3 = ctrl.Rule(driftAngle['north'] & driftSpeed['high'], (horizontalThrusters['stop'], verticalThrusters['fullAhead']))
rule4 = ctrl.Rule(driftAngle['northToNortheast'] & driftSpeed['low'], (horizontalThrusters['mediumStarboard'], verticalThrusters['lowAhead']))
rule5 = ctrl.Rule(driftAngle['northToNortheast'] & driftSpeed['medium'], (horizontalThrusters['mediumStarboard'], verticalThrusters['mediumAhead']))
rule6 = ctrl.Rule(driftAngle['northToNortheast'] & driftSpeed['high'], (horizontalThrusters['mediumStarboard'], verticalThrusters['fullAhead']))
rule7 = ctrl.Rule(driftAngle['northeastToEast'] & driftSpeed['low'], (horizontalThrusters['mediumStarboard'], verticalThrusters['lowAhead']))
rule8 = ctrl.Rule(driftAngle['northeastToEast'] & driftSpeed['medium'], (horizontalThrusters['mediumStarboard'], verticalThrusters['lowAhead']))
rule9 = ctrl.Rule(driftAngle['northeastToEast'] & driftSpeed['high'], (horizontalThrusters['fullStarboard'], verticalThrusters['lowAhead']))
rule10 = ctrl.Rule(driftAngle['east'] & driftSpeed['low'], (horizontalThrusters['mediumStarboard'], verticalThrusters['stop']))
rule11 = ctrl.Rule(driftAngle['east'] & driftSpeed['medium'], (horizontalThrusters['mediumStarboard'], verticalThrusters['stop']))
rule12 = ctrl.Rule(driftAngle['east'] & driftSpeed['high'], (horizontalThrusters['fullStarboard'], verticalThrusters['stop']))
rule13 = ctrl.Rule(driftAngle['eastToSoutheast'] & driftSpeed['low'], (horizontalThrusters['mediumStarboard'], verticalThrusters['lowAstern']))
rule14 = ctrl.Rule(driftAngle['eastToSoutheast'] & driftSpeed['medium'], (horizontalThrusters['mediumStarboard'], verticalThrusters['lowAstern']))
rule15 = ctrl.Rule(driftAngle['eastToSoutheast'] & driftSpeed['high'], (horizontalThrusters['fullStarboard'], verticalThrusters['lowAstern']))
rule16 = ctrl.Rule(driftAngle['southeastToSouth'] & driftSpeed['low'], (horizontalThrusters['mediumStarboard'], verticalThrusters['lowAstern']))
rule17 = ctrl.Rule(driftAngle['southeastToSouth'] & driftSpeed['medium'], (horizontalThrusters['mediumStarboard'], verticalThrusters['mediumAstern']))
rule18 = ctrl.Rule(driftAngle['southeastToSouth'] & driftSpeed['high'], (horizontalThrusters['mediumStarboard'], verticalThrusters['fullAstern']))
rule19 = ctrl.Rule(driftAngle['south'] & driftSpeed['low'], (horizontalThrusters['stop'], verticalThrusters['lowAstern']))
rule20 = ctrl.Rule(driftAngle['south'] & driftSpeed['medium'], (horizontalThrusters['stop'], verticalThrusters['mediumAstern']))
rule21 = ctrl.Rule(driftAngle['south'] & driftSpeed['high'], (horizontalThrusters['stop'], verticalThrusters['fullAstern']))
rule22 = ctrl.Rule(driftAngle['southToSouthwest'] & driftSpeed['low'], (horizontalThrusters['mediumPort'], verticalThrusters['lowAstern']))
rule23 = ctrl.Rule(driftAngle['southToSouthwest'] & driftSpeed['medium'], (horizontalThrusters['mediumPort'], verticalThrusters['mediumAstern']))
rule24 = ctrl.Rule(driftAngle['southToSouthwest'] & driftSpeed['high'], (horizontalThrusters['mediumPort'], verticalThrusters['fullAstern']))
rule25 = ctrl.Rule(driftAngle['southwestToWest'] & driftSpeed['low'], (horizontalThrusters['mediumPort'], verticalThrusters['lowAstern']))
rule26 = ctrl.Rule(driftAngle['southwestToWest'] & driftSpeed['medium'], (horizontalThrusters['mediumPort'], verticalThrusters['lowAstern']))
rule27 = ctrl.Rule(driftAngle['southwestToWest'] & driftSpeed['high'], (horizontalThrusters['fullPort'], verticalThrusters['lowAstern']))
rule28 = ctrl.Rule(driftAngle['west'] & driftSpeed['low'], (horizontalThrusters['mediumPort'], verticalThrusters['stop']))
rule29 = ctrl.Rule(driftAngle['west'] & driftSpeed['medium'], (horizontalThrusters['mediumPort'], verticalThrusters['stop']))
rule30 = ctrl.Rule(driftAngle['west'] & driftSpeed['high'], (horizontalThrusters['fullPort'], verticalThrusters['stop']))
rule31 = ctrl.Rule(driftAngle['westToNorthwest'] & driftSpeed['low'], (horizontalThrusters['mediumPort'], verticalThrusters['lowAhead']))
rule32 = ctrl.Rule(driftAngle['westToNorthwest'] & driftSpeed['medium'], (horizontalThrusters['mediumPort'], verticalThrusters['lowAhead']))
rule33 = ctrl.Rule(driftAngle['westToNorthwest'] & driftSpeed['high'], (horizontalThrusters['fullPort'], verticalThrusters['lowAhead']))
rule34 = ctrl.Rule(driftAngle['northwestToNorth'] & driftSpeed['low'], (horizontalThrusters['mediumPort'], verticalThrusters['lowAhead']))
rule35 = ctrl.Rule(driftAngle['northwestToNorth'] & driftSpeed['medium'], (horizontalThrusters['mediumPort'], verticalThrusters['mediumAhead']))
rule36 = ctrl.Rule(driftAngle['northwestToNorth'] & driftSpeed['high'], (horizontalThrusters['mediumPort'], verticalThrusters['mediumAhead']))
rule37 = ctrl.Rule(driftAngle['north'] | driftAngle['northToNortheast'] | driftAngle['northeastToEast']
                   | driftAngle['east'] | driftAngle['eastToSoutheast'] | driftAngle['southeastToSouth']
                   | driftAngle['south'] | driftAngle['southToSouthwest'] | driftAngle['southwestToWest']
                   | driftAngle['west'] | driftAngle['westToNorthwest'] | driftAngle['northwestToNorth']
                   & driftSpeed['nodrift'], (horizontalThrusters['stop'], verticalThrusters['stop']))
dsp_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12,
                               rule13, rule14, rule15, rule16, rule17, rule18, rule19, rule20, rule21, rule22, rule23,
                               rule24, rule25, rule26, rule27, rule28, rule29, rule30, rule31, rule32, rule33, rule34,
                               rule35, rule36, rule37])
"""
Creates a DSP control system.
"""

dsp = ctrl.ControlSystemSimulation(dsp_ctrl)
"""
``ControlSystemSimulation`` simulates the control system 
"""

"""
Simulation of our system with given drift angle and speed:
"""

angle = 182
speed = 10
dsp.input['driftAngle'] = angle
dsp.input['driftSpeed'] = speed

dsp.compute()
"""
Computes the parameters
"""

"""
Once computed, we can view the result as well as visualize it.
"""
print("Drift angle: ", angle)
print("Drift speed: ", speed)
print("Horizontal thrusters working: ", round(dsp.output['horizontalThrusters'], 3))
print("Vertical thrusters working: ", round(dsp.output['verticalThrusters'], 3))

horizontalThrusters.view(sim=dsp)
verticalThrusters.view(sim=dsp)

"""
Plots computed parameters for thrusters.
"""
plt.show()
