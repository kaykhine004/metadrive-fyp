# Autonomous Driving Policy Evaluation in MetaDrive Across Highway, Intersection, and Street Scenarios

## Abstract

This project presents a simulation-based evaluation of autonomous driving policies using the MetaDrive platform. The work focuses on three scenario categories implemented and tested in the project repository: a highway scenario, an intersection-roundabout scenario, and a street scenario with dynamic pedestrian crossings. The main purpose of the project is to compare how different autonomous driving policies perform when exposed to different types of traffic interaction and route complexity. The policies evaluated in the project are Expert Policy, Intelligent Driver Model (IDM), a hybrid Expert policy with IDM fallback, and trajectory-based control through TrajectoryIDM.

The system was implemented in Python using MetaDrive as the core simulator, with supporting libraries such as OpenCV, NumPy, Panda3D, OpenPyXL, and PyAutoGUI for visualization, route overlay, logging, and experiment export. The project includes custom scenario scripting, policy integration, repeated episode testing, result logging, and spreadsheet-based analysis. Each scenario was built with fixed route logic so that the policies could be compared under controlled and repeatable conditions. Additional scenario-specific complexity was also added. In the highway case, the ego vehicle had to cope with accident-related obstacles and mixed traffic. In the intersection case, the target vehicle had to cross a custom route composed of an entry road, a four-way intersection, a straight connector, and a roundabout while interacting with interruption traffic. In the street case, mid-route pedestrian crossings were generated dynamically on the ego vehicle's route.

The recorded results show that policy suitability is strongly scenario-dependent. In the highway scenario, IDM delivered the best overall balance of completion and safety among the saved results, while the pure Expert policy consistently failed and timed out. The hybrid Expert+IDM policy improved completion compared with pure Expert but still produced more crashes and more timeouts than plain IDM. TrajectoryIDM on the highway achieved a high completion count but accumulated an extremely large number of crashes, making it unsuitable for this scenario in its tested form. In the intersection scenario, IDM achieved the strongest overall completion performance, while TrajectoryIDM achieved good speed but higher crash counts. Expert Policy showed some successful runs in the junction environment but also suffered from low overall completion and several timeouts. In the street scenario, the saved ten-episode Expert run completed all episodes without crashes, and the available IDM street outputs also completed successfully without recorded crashes.

Overall, the project demonstrates that there is no single policy that is equally strong in every environment. Instead, the performance of a driving policy depends heavily on the structure of the route, the density and behavior of surrounding traffic, and the presence of dynamic obstacles such as pedestrians. The project therefore provides a complete experimental framework, based entirely on the implemented work and recorded outputs, for comparing autonomous driving strategies in MetaDrive.

## Chapter 1 Introduction

### 1.1 Project Background

Autonomous driving is an interdisciplinary field that combines vehicle dynamics, control, route planning, navigation, interaction handling, and safety management. A driving agent must make decisions continuously while operating in an environment that changes from moment to moment. These changes may come from road layout, vehicle density, lane-level constraints, static obstacles, and moving traffic participants such as cars and pedestrians. Because of this complexity, simulation has become a major tool for autonomous driving development and evaluation.

The project presented in this report was developed around MetaDrive, an open-ended driving simulator that supports configurable road structures, built-in vehicle policies, single-agent and multi-agent environments, route-based navigation, and custom scenario design. MetaDrive is suitable for this project because it allows the developer to move beyond static demonstrations and create repeatable experimental conditions in which multiple control strategies can be compared across several road contexts.

The work in this project is centered on the comparison of autonomous driving methods in three different categories of driving environment. The first category is a highway scenario in which the ego vehicle drives through a structured route containing highway-like road segments, mixed traffic, and accident-related obstacles. The second category is a multi-agent intersection scenario in which a target vehicle must travel along a custom fixed path from an entry road through a four-way intersection and then into a roundabout while interacting with surrounding traffic vehicles. The third category is a street scenario in which the ego vehicle follows a fixed route in a lower-speed urban environment and must deal with dynamically spawned pedestrian crossing events.

The motivation behind the project comes from a practical observation made during implementation and testing: a policy that appears strong in one scenario may perform poorly in another. The behavior of built-in MetaDrive policies, and of custom variants created in this project, changes significantly when the scenario shifts from relatively structured highway driving to conflict-heavy intersection driving or to street driving with pedestrians. This means that policy evaluation cannot be treated as a single-scenario problem. Instead, the robustness and suitability of a policy must be judged with respect to the environment in which it is used.

The project therefore focuses on building a scenario-driven evaluation pipeline rather than only presenting isolated demonstrations. It includes custom map generation, route locking, dynamic object spawning, traffic control, on-screen monitoring, multi-episode testing, and spreadsheet export for analysis. In this sense, the project is not only about running MetaDrive examples. It is about adapting the simulator to specific research questions and using the recorded outcomes to compare rule-based, expert-based, hybrid, and trajectory-based driving methods.

### 1.2 Problem Statement

Autonomous driving control policies do not behave uniformly across all environments. A method that can follow a route successfully in a controlled environment may fail when traffic becomes denser, when road geometry becomes more complex, or when non-vehicle agents such as pedestrians are introduced. For a project that aims to compare autonomous driving strategies meaningfully, this creates a clear problem: it is not enough to know whether a policy can drive; it is necessary to know where it drives well, where it fails, and why.

During the implementation of this project, the built-in and modified policies produced noticeably different results across scenarios. The pure Expert Policy, which might be expected to perform strongly, was unable to handle the tested highway configuration and repeatedly timed out. IDM, which is simpler and more reactive, performed much better on that scenario. In the intersection environment, Expert Policy was capable of some successful runs, but its completion performance remained substantially below IDM. In the street environment, both Expert and IDM showed clean behavior in the saved result files, but the available IDM street result exports are fewer than the other compiled scenario outputs. The trajectory-based controllers also displayed mixed performance, showing strong movement efficiency in some cases but much higher crash frequency in others.

The core problem addressed by this project is therefore the lack of a consistent, project-specific framework for comparing autonomous driving policies across multiple MetaDrive scenarios that represent different traffic and route conditions. Without such a framework, it is difficult to determine which policy should be preferred for a given environment, whether policy extensions such as IDM fallback are helpful, and how trajectory-following behavior compares with more reactive control.

This project addresses that problem by implementing three scenario classes, integrating multiple policy variants, defining common logging metrics, and running repeated experiments that allow direct comparison. The report then interprets those results using only the implemented code and the saved experiment outputs produced by the project.

### 1.3 Project Objectives

The overall objective of this project is to design, implement, and evaluate an autonomous driving simulation framework in MetaDrive that can compare different control policies across highway, intersection, and street scenarios.

The specific objectives of the project are:

1. To build and configure three autonomous driving simulation scenarios in MetaDrive that reflect different classes of road interaction.
2. To implement and test several driving policies, including Expert Policy, IDM Policy, Expert Policy with IDM fallback, and TrajectoryIDM-based control.
3. To create custom scenario logic where needed, such as route-constrained multi-agent intersection behavior and mid-route pedestrian crossing in the street environment.
4. To log performance outputs systematically for repeated episodes and export the results into analysis-friendly spreadsheet format.
5. To compare the policies based on completion, timeout, crash count, travel efficiency, and average speed.
6. To identify which policy is most suitable for each implemented scenario and to explain the observed differences using the project’s own results.

### 1.4 Scope of Work

This project is limited to simulation-based autonomous driving evaluation. It does not involve physical vehicles, real sensors, real road deployment, or hardware-in-the-loop integration. The work is entirely contained within the MetaDrive simulation environment and the Python project repository used to build and test the scenarios.

The scope includes three implemented scenario families. The highway scenario uses SafeMetaDriveEnv and a fixed route string to generate a highway-like path with traffic, road events, and accident-related obstacles. The intersection scenario uses MultiAgentMetaDrive and a custom-generated map consisting of four main road blocks: entry straight, intersection, connector straight, and roundabout. The street scenario uses MetaDriveEnv with a fixed block sequence and introduces dynamic pedestrian crossing events in the middle portion of the route.

The scope also includes the implementation of policy control logic and policy comparison. The evaluated policies are limited to those that were implemented and tested in the project scripts and result files. These include Expert Policy, IDM Policy, StuckAware Expert with IDM fallback, TrajectoryIDM-based control, and the multi-agent route-specific TrajectoryIDM variant in the intersection scenario.

The project further covers visualization, route overlay, logging, and result export. It does not cover perception model training, camera-based lane detection, sensor fusion, map localization, reinforcement learning training pipelines, or real-world calibration. Reinforcement learning appears in the conceptual discussion because MetaDrive is commonly associated with RL research, but the actual experiments performed in this project are based on built-in or modified driving policies rather than trained RL agents.

### 1.5 Significance of the Project

The significance of this project lies in its practical comparison of autonomous driving policies under several driving situations rather than under a single example environment. In many student projects or early simulator studies, a controller is tested only in one simple route, making it difficult to judge whether the result reflects actual robustness or only a good match between the policy and a specific environment. This project reduces that problem by testing the policies across three clearly different contexts.

The work is also significant because it includes both use of standard MetaDrive mechanisms and direct customization of the environment. Instead of treating MetaDrive as a closed simulator, the project uses custom map managers, spawn managers, route creation, multi-agent policy assignment, dynamic pedestrian control, on-screen route overlays, and per-episode result logging. This makes the work a complete autonomous driving evaluation framework rather than a set of disconnected tests.

Finally, the significance of the project is visible in the results themselves. The experiments show that an apparently stronger policy can fail badly in one environment while a simpler reactive policy can outperform it. These findings are important because they demonstrate that autonomous driving evaluation should be context-sensitive and evidence-driven.

### 1.6 Report Structure

This report is organized into seven chapters.

Chapter 1 introduces the project, including its background, problem statement, objectives, scope, significance, and structure.

Chapter 2 presents the literature review. In this report, that chapter is written strictly around the technical foundations actually used in the project, namely simulation-based driving, MetaDrive scenario construction, rule-based driving, expert and trajectory-based control, and the motivation for comparing these methods.

Chapter 3 explains the project methodology, including the workflow, tools, environments, metrics, and experiment design.

Chapter 4 describes the overall system design and implementation, including the project architecture, the three main scenarios, the implemented policies, and the result export mechanism.

Chapter 5 details the experimental setup, covering the hardware and software environment, parameter configuration, compared baselines, episode counts, and evaluation criteria.

Chapter 6 presents the recorded experimental results and discusses the observed behavior of each policy in each scenario.

Chapter 7 concludes the report by summarizing the major findings of the project.

## Chapter 2 Literature Review

### 2.1 Autonomous Driving in Simulation

In the context of this project, autonomous driving in simulation refers to the design and testing of vehicle control behavior inside a software environment that reproduces roads, traffic, and driving interactions. Simulation is especially appropriate for this project because the work depends on repeated comparison. The same scenario must be run multiple times with different policies and with consistent evaluation logic. Achieving this level of repeatability in the real world would be difficult, expensive, and potentially unsafe.

The use of simulation in this project is not limited to visual demonstration. It supports the full experiment cycle. The simulator produces the road network, traffic participants, and ego vehicle environment. The driving policies take actions at every environment step. The simulation state updates continuously and provides route completion, crash information, and step counts. The system then logs these values and exports them for analysis. In this way, simulation is the core experimental platform rather than only a visualization tool.

Another reason simulation is important in this project is the diversity of tested environments. The highway scenario includes relatively high-speed driving and obstacle interaction. The intersection scenario requires navigation through a junction-heavy custom map with other vehicles. The street scenario adds pedestrian interaction. Testing all of these in a physical environment would require significant infrastructure, repeated safety setup, and carefully controlled conditions. MetaDrive allows all of these conditions to be reproduced inside the same software project.

The project also shows that simulation is useful not only for success confirmation but for failure diagnosis. For example, the highway result files make it clear that the pure Expert Policy does not merely produce occasional mistakes in that environment. It fails systematically, with repeated timeout behavior and extremely low average speed. Such patterns are easier to observe and compare when the scenario is simulated and logged consistently.

### 2.2 MetaDrive Environment and Relevance

MetaDrive is central to this project because it offers the flexibility required for both quick configuration and custom environment construction. The project uses three main MetaDrive environment classes: `SafeMetaDriveEnv` for the highway scenario, `MultiAgentMetaDrive` for the intersection-roundabout scenario, and `MetaDriveEnv` for the street scenario. This already shows one of MetaDrive’s strengths: the same simulator ecosystem can support different scales of driving experiment.

The project also uses built-in MetaDrive policies such as `ExpertPolicy`, `IDMPolicy`, and `TrajectoryIDMPolicy`. These policies provide a consistent starting point for comparison. However, the project does not stop at direct usage. It modifies or extends them where necessary. For instance, the highway script defines `StuckAwareExpertPolicy`, which monitors speed and movement over time and switches to IDM if the vehicle is effectively stuck. The highway trajectory script defines an `ObstacleAwareTrajectoryIDMPolicy` that can switch into an avoidance mode when progress stalls. These changes show that MetaDrive is not only a simulator to run prebuilt code. It is also a development framework in which policy behavior can be adapted for project-specific testing.

MetaDrive is also relevant because of the way it exposes map and route structures. In the intersection scenario, the project directly constructs a custom road network from `FirstPGBlock`, `InterSection`, `Straight`, and `Roundabout` blocks. A custom map manager and spawn manager are then used to force a fixed route for the target agent and separate behavior for interruption vehicles. This kind of route-level control is essential for fair policy comparison because it reduces the influence of uncontrolled destination changes.

The project’s use of rendering and top-down overlays further demonstrates MetaDrive’s relevance. Several scripts draw start and end markers, show a live agent icon, and present route progress or policy status on screen. These functions are not the main evaluation outputs, but they greatly improve interpretability during testing.

### 2.3 Rule-Based Driving Methods

Rule-based driving methods are an important foundation for this project because they are interpretable and easy to compare. A rule-based policy makes decisions according to programmed logic instead of learning them through data-driven training. This is useful in a project where the main objective is to evaluate behavior under controlled conditions rather than to train a model from scratch.

The most important rule-based method in this project is IDM. IDM regulates speed by reacting to the state of traffic ahead. In practical terms, this means that the vehicle accelerates, decelerates, and maintains safe following behavior according to surrounding conditions. In the project scripts, IDM serves both as a baseline policy on its own and as a fallback or component within hybrid policies. This is a strong indication of its practical value in the implemented work.

Rule-based methods are especially useful when the evaluation criteria are safety and stability. Since their behavior is explicitly designed, it is easier to relate output patterns back to policy design. For example, the good highway completion results from IDM can be interpreted as evidence that reactive traffic-following behavior matches the highway scenario more effectively than the project’s tested version of Expert Policy. Similarly, the crash-heavy highway performance of TrajectoryIDM suggests that following a route aggressively without enough environment-specific adaptation can be harmful in that context.

A further advantage of rule-based methods in this project is that they allow controlled extension. The project does not need to replace a working policy completely to improve it. It can add fallback logic, switching conditions, or route-specific assignment while keeping the base rule-driven behavior understandable.

### 2.4 Expert Policy, IDM, and Trajectory-Based Control

The project compares three broad styles of autonomous driving control: Expert Policy, IDM-based control, and trajectory-based control.

Expert Policy in this project acts as a high-level built-in controller intended to drive the agent toward its destination while respecting the route and reacting to the environment. In the street and some intersection runs, it is able to complete the route successfully. However, the recorded highway results show that Expert Policy is not universally reliable. The highway workbook identifies it as a failed baseline in that scenario, and the saved episode records show zero passes and repeated timeouts. This is one of the most important findings of the project because it shows that a stronger nominal policy can still become a weak baseline when the scenario does not match its operational strengths.

IDM is the most consistently stable rule-based baseline in the project. It is used directly in the highway, intersection, and street scenarios. The recorded results show that it achieves the best overall completion rate in both the highway and intersection compiled workbooks. This does not mean that IDM is perfect, because it still accumulates crashes in some episodes. However, it shows that reactive control is highly competitive across the implemented scenarios.

Trajectory-based control introduces explicit route following by constructing a `PointLane` path and using `TrajectoryIDMPolicy` to guide the ego vehicle along it. This is attractive because it gives the controller a clear path structure to follow, which is useful in custom route environments like the intersection scenario. However, the project results also show that trajectory following alone does not guarantee safe operation. In the highway case, the saved result file shows very high crash frequency despite high completion. This means that completion must not be interpreted as the only success criterion. A trajectory controller can still be unsuitable if it reaches the goal while colliding too often.

The project also explores hybrid behavior. `StuckAwareExpertPolicy` uses Expert Policy first and then transfers control to IDM if signs of obstruction appear. This is important because it reflects a practical engineering approach. Instead of discarding a policy completely, the project adds a recovery mechanism that can handle cases where the original controller is weak.

### 2.5 Reinforcement Learning for Driving Tasks

Although reinforcement learning is not the main experimental method in this project, it forms part of the conceptual background because MetaDrive is widely associated with learning-based driving research. Reinforcement learning differs from the implemented policies in this project because it would require training through reward optimization rather than direct use of built-in rule-based or hybrid controllers.

The relevance of reinforcement learning here is mainly methodological. If a future extension of the project were to introduce a trained policy, the existing highway, intersection, and street scenarios would already provide a useful evaluation benchmark. The recorded IDM, Expert, and TrajectoryIDM results establish concrete performance references for pass rate, timeout behavior, crash count, speed, and travel time. In this sense, the current project creates a baseline framework that is compatible with later learning-based comparison, even though the current report remains strictly focused on the implemented non-RL policies.

There is also a more direct lesson. The project results show that good autonomous driving behavior requires adaptation to scenario structure. This is exactly the kind of challenge that motivates learning-based approaches. However, the project also shows that robust rule-based and hybrid baselines can be very strong. Therefore, any future RL comparison would need to outperform these established baselines in a meaningful way rather than assuming that learned control is automatically better.

### 2.6 Research Gap and Motivation

The research gap addressed by this project is not a missing theory of autonomous driving. Instead, it is a missing project-level comparison across multiple MetaDrive scenarios that are all implemented, controlled, logged, and analyzed within one framework. Many driving experiments focus on a single environment or only demonstrate one policy. That kind of setup is not sufficient to answer the main question of this project, which is how different policies behave when the environment changes.

The motivation for the project came directly from observed behavior during testing. The highway scripts revealed that Expert Policy could fail repeatedly in the tested highway map. This made it clear that the policy comparison needed to include scenario variation and fallback design. The custom intersection work then provided a second environment in which the relationship between policy type and performance could be tested under route conflict. The street scenario added a third environment with pedestrian interactions, making the project more complete and more representative of different driving demands.

As a result, the project is motivated by practical engineering need. The question is not simply which policy works. The question is which policy works where, under what conditions, and with what trade-offs in completion, safety, and efficiency.

## Chapter 3 Project Methodology

### 3.1 Overall Workflow

The workflow of the project began with simulator preparation and environment selection. The MetaDrive repository was used as the base simulator platform. The project then organized the work under a `Final analysis` structure, separating the implementations and outputs for highway, intersection, and street scenarios. This organization helped keep the development process scenario-focused and made it easier to compare outputs later.

The second stage of the workflow was scenario implementation. The highway scenario scripts configured SafeMetaDrive with a fixed route string, defined the ego spawn lane, enabled traffic and accident-related settings, and added route overlay visualization. The intersection scenario required more involved environment design. A custom map was built from MetaDrive road blocks, and the route of the target vehicle was fixed using custom managers. The street scenario was then implemented with a separate fixed route and a custom mid-route pedestrian controller that created crossing events according to route progress and vehicle proximity.

The third stage was policy integration. Each relevant policy was assigned to the ego vehicle depending on the script being tested. In some cases, such as the intersection scenario, the policy assignment also distinguished between the target vehicle and surrounding traffic vehicles. In the highway hybrid and trajectory variants, policy behavior was modified to recover from stuck states or switch to avoidance mode.

The fourth stage was execution and monitoring. The scripts ran single episodes for visual inspection and multi-episode runs for result collection. During execution, the simulator displayed driving information such as route completion, speed, crash count, and simulation time. In some scripts, top-down route views and start-end markers were also shown inside the simulation window.

The final stage was logging and analysis. The episode outputs were written into Excel workbooks or CSV files. The workbooks were then used to compile repeated-run results and calculate pass counts, timeout counts, crash frequency, average speed, and time-to-goal patterns. These outputs form the basis of the results chapter in this report.

### 3.2 Software Tools and Framework

The primary programming language used in the project is Python. The main framework is MetaDrive, which provides the vehicle simulation, road generation, environment stepping, traffic interaction, route navigation, and built-in policy classes used in the experiments.

Panda3D is part of the rendering stack used by MetaDrive and supports the 3D simulation window and texture overlay functions. OpenCV and NumPy are used in several scripts for building top-down route images, drawing start and end markers, showing the tracked vehicle on the route overlay, and updating map textures frame by frame. These libraries are particularly visible in the highway and intersection scripts where a live map is drawn in the same window as the main simulation.

OpenPyXL is used for exporting repeated episode results to Excel format. The street scripts also include a CSV fallback path so that result export can still succeed when spreadsheet dependencies are unavailable. PyAutoGUI is used in rendered runs to send the `F` key and activate unlimited frame rate mode where supported, making long experiments faster to complete.

The project setup also includes additional packages declared in the repository, such as Matplotlib, Pandas, SciPy, Pillow, TQDM, and Pygame. While not every declared dependency is visible in every script, the overall environment is prepared as a full simulation-capable Python workspace rather than a minimal one-script demo.

### 3.3 Simulation Environments Used

The project uses three categories of simulation environment.

The highway environment is built on `SafeMetaDriveEnv`. It uses a fixed map string, fixed spawn lane, mixed traffic, inverse traffic, and accident-related settings. The highway is intended to test how well the control policies handle faster movement, traffic flow, and obstacle interaction. It is the most important scenario for exposing the weakness of pure Expert Policy in the recorded outputs.

The intersection environment is built on `MultiAgentMetaDrive`. It is not a standard off-the-shelf scenario. Instead, it constructs a custom route from a sequence of road blocks. The target vehicle starts from a fixed entry road and aims for a fixed roundabout exit. Other vehicles appear from interruption roads and are routed independently. This design makes the scenario structured enough for fair comparison while still containing meaningful traffic conflict.

The street environment is built on `MetaDriveEnv` with a block-sequence map configuration and a deterministic spawn lane. Unlike the highway and intersection cases, the street scenario introduces pedestrians as dynamic traffic participants using a dedicated controller. These pedestrians are spawned mid-route and cross the ego vehicle’s lane according to trigger conditions. This makes the street environment the most explicit test of vehicle interaction with non-vehicle participants.

### 3.4 Performance Metrics

The project uses several metrics to compare policy behavior. These metrics are extracted directly from the saved workbooks and from the logic inside the scenario scripts.

`Pass` indicates whether the episode satisfied the success logic defined in the script. In the highway and intersection compiled workbooks, pass status reflects whether the run was considered successful overall. In the street scripts, pass is tied closely to the absence of timeout conditions.

`Timeout` indicates whether the policy failed to maintain sufficient progress. In the street scripts, timeout is determined by a stall condition based on low speed sustained for more than sixty seconds of simulated time. In the other scenarios, timeout information is present in the result sheets and is used to distinguish non-progress failures from successful completion.

`Crash Count` is a direct safety measure. The scripts detect crashes by reading crash flags such as vehicle collision, object collision, and in the street environment human collision. Some scripts count only edge-triggered changes so that a continuous collision state is not counted as repeated separate crashes unless the state changes again.

`Time Taken to Reach Goal` measures travel efficiency. In the highway and intersection compiled workbooks this is recorded as a travel-duration value. In the street ten-episode scripts, the workbook includes both `time_to_goal_s` and `episode_sim_time_s`. The former is based on measured execution time and the latter on simulated time derived from the physics update rate. This means the street scenario provides both wall-time completion and simulated-time duration information.

`Average Speed` provides a compact indication of movement quality and flow efficiency. A very low value usually indicates blocking, repeated stopping, or poor adaptability, while a very high value must still be interpreted alongside crash count because fast movement alone does not imply safe driving.

### 3.5 Experimental Design and Comparison Strategy

The experimental design combines scenario-specific implementation with repeated policy evaluation. The project does not compare all policies in a single universal environment. Instead, it tests policies in the environment types for which they were actually implemented and saved in the repository.

For the highway scenario, the compared methods are IDM, Expert, Expert+IDM, and TrajectoryIDM. The highway workbook contains repeated runs for each policy, allowing comparison based on pass rate, timeout rate, crash average, average speed, and goal time.

For the intersection scenario, the compared methods are IDM, Expert, and TrajectoryIDM. These are all tested on the same custom route environment with a tracked target agent and interruption traffic.

For the street scenario, the most complete saved multi-episode result set is the ten-episode Expert workbook. IDM also has saved outputs in the result directory, but fewer rows are currently available in the saved files. Since this report must remain strictly based on existing project outputs, the report uses the available saved street files as they are rather than assuming missing episodes.

The comparison strategy follows a balanced interpretation principle. A policy is not judged by completion alone. A result with high pass count but very high crash count is treated differently from a result with slightly lower pass count but far stronger safety behavior. This is especially important in the highway scenario, where TrajectoryIDM reaches the goal often but does so with an unacceptably high crash average.

## Chapter 4 System Design and Implementation

### 4.1 Overall Architecture

The architecture of the project can be understood as a four-layer system.

The first layer is the scenario layer. This layer defines the road environment, map structure, spawn positions, destination logic, traffic density, and dynamic event behavior such as pedestrians. Each of the three main scenario families has its own implementation in this layer.

The second layer is the policy layer. Here, the ego vehicle is assigned a control strategy. This may be a direct built-in policy such as Expert or IDM, or a modified version such as `StuckAwareExpertPolicy` or `ObstacleAwareTrajectoryIDMPolicy`. In the multi-agent intersection environment, this layer also includes logic that assigns different policies to different agents.

The third layer is the execution and monitoring layer. This layer steps the environment, updates the scenario state, computes dynamic behaviors such as crossing pedestrians, detects crash transitions, and renders both the 3D simulation and the top-down overlays.

The fourth layer is the logging layer. This layer accumulates per-episode results and exports them into Excel or CSV format. Since the project depends heavily on result comparison, this final layer is not optional. It is a core part of the implementation.

The architectural separation is important because it makes the project extendable. Each scenario uses the same general flow of environment setup, policy assignment, execution, monitoring, and result export, but the scenario logic itself remains independent. This makes the framework coherent while preserving scenario-specific behavior.

### 4.2 Highway Scenario Implementation

The highway scenario uses `SafeMetaDriveEnv` with a fixed route string:

`CSrRSY$yRSCR`

This route is described inside the project scripts as a stable path containing highway-relevant components such as ramps, splits or merges, and tollgate-related sections. The ego vehicle always spawns on the same lane using `FirstPGBlock.NODE_2` to `FirstPGBlock.NODE_3`. This deterministic spawn placement reduces variability and allows the policy comparison to focus on control behavior rather than on route initialization differences.

Traffic density in the highway scenario is configured at `0.05`, with `need_inverse_traffic=True` and `random_traffic=True`. Accident probability is set to `0.25`, and static traffic objects are enabled in the highway scripts. The horizon is set to `5000`. These settings collectively create a scenario that is relatively structured in route geometry but rich in interaction challenges.

The highway implementation also includes route visualization. The code builds a top-down map surface, extracts the start and end positions from the navigation data, draws colored markers, and shows a live agent arrow on the overlay. This makes it possible to track the policy’s path and visually inspect when and where failures occur.

Four policy variants are present in the highway analysis work:

1. `highway_Expert.py` uses pure `ExpertPolicy`.
2. `highway_IDM.py` uses pure `IDMPolicy`.
3. `highway_Expert+IDM.py` uses a custom `StuckAwareExpertPolicy`.
4. `highway_Trajectory.py` uses a custom obstacle-aware trajectory policy based on `TrajectoryIDMPolicy`.

The `StuckAwareExpertPolicy` is one of the most important custom implementations in the project. It monitors the vehicle’s speed and movement in two stages. A fast check triggers when speed falls below `8.0 km/h` for ten consecutive steps. A slower fallback check compares positional movement over a thirty-step window and triggers if the vehicle has moved less than `3.0 m`. When either condition is met, the controller switches from Expert behavior to IDM behavior. This modification was designed specifically to solve the obstacle-handling weakness observed in the highway scenario.

The obstacle-aware trajectory controller is also scenario-specific. It follows a pre-built `PointLane` trajectory and uses IDM speed control, but can switch into lane-change mode when the vehicle remains stuck below a low speed threshold. It then returns to trajectory mode after sustained recovery. This is a more complex controller than plain IDM or Expert because it tries to combine planned route following with reactive obstacle avoidance.

### 4.3 Intersection Scenario Implementation

The intersection scenario is implemented as a custom multi-agent environment. This is one of the most technically detailed parts of the project because the route is constructed directly from MetaDrive building blocks.

The map is built as follows:

1. `FirstPGBlock` creates the entry straight road and provides the spawn road for the target vehicle.
2. `InterSection` creates a four-way junction.
3. `Straight` creates a connector road leading away from the intersection.
4. `Roundabout` creates the final circular junction and destination region.

The target vehicle is fixed as `agent0`. It always spawns on the same entry road and always aims for the same roundabout exit node. This route locking is essential because it ensures that policy comparisons are based on the same navigation task.

To support this structure, the project defines:

1. A custom `RoundaboutIntersectionMap` class to generate the road network.
2. A custom `RoundaboutIntersectionMapManager` to load the map and expose the correct spawn roads.
3. A custom `RoundaboutIntersectionSpawnManager` to keep the target route fixed while assigning surrounding vehicles to alternative destinations.
4. In the trajectory-based version, a custom `TrajectoryIDMAgentManager` that gives `TrajectoryIDMPolicy` to the target agent and `IDMPolicy` to the other agents.

The environment contains ten agents in total. The tracked vehicle is the focus of evaluation, but the surrounding vehicles create the interruptions that make the scenario meaningful. The top-down overlay logic is again used here. The scripts render the 3D view, then render an off-screen top-down image, draw start, end, and vehicle markers, and display the resulting texture in the render window.

This scenario is important because it tests route adherence and interaction handling at the same time. The target vehicle must follow a fixed path, but it cannot do so in isolation. The interruption traffic changes the timing of available gaps and forces the controller to respond dynamically.

### 4.4 Street Scenario Implementation

The street scenario uses `MetaDriveEnv` with a block-sequence map configuration:

`SCTSyCP`

The map is configured with `lane_num=1`, fixed spawn lane selection, enabled crosswalk display, and moderate traffic density. This creates a narrower, more urban driving context than the highway scripts.

The most important custom feature in the street scenario is the `MidRouteCrossingController`. This controller is responsible for selecting a crossing point near the middle of the route and spawning pedestrians that cross the ego vehicle’s lane when trigger conditions are met. The implementation does not simply place pedestrians at the beginning of the episode. Instead, it waits until route progress reaches a trigger range and then resolves a crossing location either on the ego lane itself or through a graph-based fallback. This makes the pedestrian event appear naturally during the drive.

In the ten-episode Expert street script, the pedestrian count range is `1-2` per episode. In the IDM street script, the range is `2-4` per episode. The controller also applies randomization to route progress trigger, approach distance, curb wait steps, pedestrian walking speed, and spawn side. These details increase behavioral variety while keeping the scenario logic structured.

Another practical design feature is the cleanup of spawned pedestrian objects before each environment reset. This was necessary because multi-episode runs would otherwise accumulate dynamic objects and could trigger assertions or physics world conflicts. The explicit cleanup routine is therefore an important part of making repeated street experiments stable.

The street ten-episode scripts also contain a clear stall definition. If the vehicle remains below `1.0 km/h` for `60.0` seconds of simulated time, the run is marked as timed out. Simulated time per environment step is derived directly from `physics_world_step_size` and `decision_repeat`, ensuring that timeout logic matches the actual physics integration used by MetaDrive.

### 4.5 Policy Implementations

The policies tested in this project are closely tied to the scenario logic.

`ExpertPolicy` is used in the highway, intersection, and street scenarios. In the highway environment, the raw recorded results show that it is not robust enough in the tested configuration. In the intersection environment, it produces some successful runs but lower completion than IDM. In the street environment, the saved ten-episode result file shows clean, fully successful runs under the tested settings.

`IDMPolicy` is the most direct rule-based baseline and is used in all three scenario categories. It does not require custom route point construction and instead relies on reactive car-following and navigation behavior provided through MetaDrive.

`StuckAwareExpertPolicy` is a project-specific hybrid controller that extends `ExpertPolicy`. It is only used in the highway hybrid script and is one of the clearest examples of engineering adaptation in the project. The underlying design idea is simple: keep Expert behavior when it works, but transfer control to IDM when signs of obstruction appear.

`TrajectoryIDMPolicy` is used in both highway and intersection trajectory scripts. In the highway case, it is extended through an obstacle-aware controller that supports lane-change avoidance mode. In the intersection case, it is assigned only to the target agent while the other agents remain on IDM. This is important because the target route is fixed and suitable for explicit trajectory following, while the interruption vehicles are better treated as traffic agents rather than as route-tracked experimental subjects.

The project therefore does not compare abstract policy names only. It compares concrete implementations tied to the designed scenarios.

### 4.6 Data Logging and Result Export

The logging design is consistent across the project. Each repeated-run script records episode-level outputs into a list of dictionaries and exports them after the specified number of episodes. The exported fields vary slightly by script, but the core result structure remains the same: episode identifier, pass status, timeout status, crash count, time to goal, average speed, and date-time stamp.

The street scripts save results into dedicated `expert analysis` and `IDM analysis` folders. The highway and intersection work include both individual result files and compiled analysis workbooks. The compiled workbooks are especially useful because they already organize the repeated results by policy category and include notes on baseline status, including the project observation that pure Expert failed in the highway scenario and that the trajectory policy was not suitable for that scenario.

The export implementation is practical and resilient. If `openpyxl` is not available, some scripts automatically switch to CSV export so that testing is not blocked. This design choice supports reliable experimentation even when the Python environment changes.

## Chapter 5 Experimental Setup

### 5.1 Hardware and Software Setup

The project was developed and executed in a local Windows-based development environment. This can be seen from the absolute file paths used in the scenario scripts and result directories. The project runs inside a Python environment configured for MetaDrive, with the simulator and supporting libraries installed in the repository workspace.

The software stack includes:

1. Python as the main implementation language.
2. MetaDrive Simulator as the driving simulation framework.
3. Panda3D for rendering and simulation window support.
4. NumPy and OpenCV for top-down image generation and overlay drawing.
5. OpenPyXL for spreadsheet export.
6. PyAutoGUI for frame-rate toggle automation in rendered runs.

The repository setup also declares additional utility packages such as Matplotlib, Pandas, SciPy, Pillow, Pygame, TQDM, Requests, and others. These form the complete environment needed to build, run, and analyze the project.

### 5.2 Simulation Parameters

The main simulation parameters differ by scenario.

For the highway scenario:

1. Environment type: `SafeMetaDriveEnv`
2. Map string: `CSrRSY$yRSCR`
3. Start seed: `5`
4. Number of scenarios: `1`
5. Traffic density: `0.05`
6. Inverse traffic: enabled
7. Random traffic: enabled
8. Accident probability: `0.25`
9. Static traffic object: enabled
10. Horizon: `5000`
11. Fixed ego spawn lane: `(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 0)`

For the intersection scenario:

1. Environment type: `MultiAgentMetaDrive`
2. Number of agents: `10`
3. Horizon: `10000`
4. Fixed target spawn road: `Road(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3)`
5. Fixed target destination node: `Roundabout.node(3, 1, 3)`
6. Map config: `exit_length=60`, `lane_num=2`
7. Top-down camera coordinates set for stable route visualization

For the street scenario:

1. Environment type: `MetaDriveEnv`
2. Map config type: `block_sequence`
3. Map string: `SCTSyCP`
4. Lane count: `1`
5. Start seed: `0`
6. Number of scenarios: `1`
7. Traffic density: `0.06`
8. Random traffic: enabled
9. Inverse traffic: enabled
10. Accident probability: `0.0`
11. Horizon: `8000`
12. Crosswalk display: enabled
13. Fixed ego spawn lane: `(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 0)`

In addition to these fixed values, the street scenario contains randomized crossing parameters such as trigger route progress, approach distance, pedestrian waiting time, pedestrian speed, and pedestrian count.

### 5.3 Baseline Methods Compared

The compared methods are taken directly from the implemented scripts and recorded workbook outputs.

In the highway scenario, the baseline set includes:

1. IDM Policy
2. Expert Policy
3. Expert Policy with IDM fallback
4. TrajectoryIDM-based control with obstacle-aware logic

In the intersection scenario, the baseline set includes:

1. IDM Policy
2. Expert Policy
3. TrajectoryIDM Policy for the target agent with IDM for the surrounding agents

In the street scenario, the saved evaluation outputs include:

1. Expert Policy
2. IDM Policy

Although a street trajectory-related script exists in the project folder, the current saved street result files available in the analysis folders are the Expert and IDM outputs. Therefore, the final report keeps the street results strictly within those saved records.

### 5.4 Number of Episodes and Test Conditions

The experiment design uses repeated episodes to make the outputs meaningful.

The highway compiled workbook contains:

1. Fifty saved episodes for IDM
2. Ten saved episodes for Expert
3. Fifty saved episodes for Expert+IDM
4. Ten saved episodes for TrajectoryIDM

The intersection compiled workbook contains:

1. Fifty saved episodes for IDM
2. Fifty saved episodes for Expert
3. Fifty saved episodes for TrajectoryIDM

The street analysis folders contain:

1. A complete ten-episode saved Expert workbook
2. Additional partial saved Expert output from a separate run
3. A small number of saved IDM rows across two workbooks

The test conditions kept the route fixed inside each scenario category while allowing traffic interaction and dynamic events to occur. This means that the scenario task was repeatable, but the policies still had to respond to realistic variation in traffic timing and in the randomized parts of the street pedestrian controller.

### 5.5 Evaluation Criteria

The main evaluation criteria are:

1. Success of task completion
2. Ability to avoid timeout or stall failure
3. Low crash frequency
4. Efficient travel time
5. Stable average speed

These criteria were selected because they reflect the actual goals of the project. A policy should not only reach the goal. It should do so with reasonable safety and without becoming stuck. Likewise, a policy that moves quickly but produces excessive collisions should not be considered strong overall.

### 5.6 Preparation of Result Analysis

The project’s result analysis is based on the saved Excel workbooks located under the `Final analysis` directories. For highway and intersection, the project already provides compiled workbooks that aggregate repeated episodes by policy category. For the street scenario, the most complete saved result set is the ten-episode Expert workbook, supplemented by the available IDM result files. The analysis in the next chapter is therefore based entirely on the outputs already produced by the project scripts and not on any externally reconstructed data.

## Chapter 6 Results and Discussion

### 6.1 Overview of Saved Results

The saved results demonstrate that policy performance is highly dependent on scenario structure. This is the central finding of the project. No policy is universally dominant across all scenarios in every metric. Instead, each policy exhibits a recognizable pattern of strengths and weaknesses.

Across the compiled workbooks, IDM shows the strongest overall consistency. It achieves the highest completion results in both the highway and intersection environments and maintains competitive speed. However, it is not always the lowest-crash policy. The trajectory-based policies often move faster but can accumulate many more collisions. The pure Expert policy is highly sensitive to scenario type, performing very poorly in the tested highway configuration but achieving some successful runs in the intersection environment and a clean ten-episode result in the street environment.

The saved results are summarized below.

**Highway scenario**

| Policy | Episodes | Pass Rate | Timeout Rate | Avg. Crash Count | Avg. Speed (km/h) | Avg. Goal Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| IDM | 50 | 98.0% | 2.0% | 1.04 | 24.52 | 165.12 |
| Expert | 10 | 0.0% | 100.0% | 1.40 | 3.75 | 120.00 |
| Expert+IDM | 50 | 82.0% | 18.0% | 3.52 | 21.13 | 169.44 |
| TrajectoryIDM | 10 | 90.0% | 10.0% | 35.40 | 23.00 | 188.90 |

**Intersection scenario**

| Policy | Episodes | Pass Rate | Timeout Rate | Avg. Crash Count | Avg. Speed (km/h) | Avg. Goal Time (s) |
|---|---:|---:|---:|---:|---:|---:|
| IDM | 50 | 100.0% | 0.0% | 2.82 | 26.11 | 50.60 |
| Expert | 50 | 38.0% | 14.0% | 1.47 | 15.12 | 59.95 |
| TrajectoryIDM | 50 | 72.0% | 0.0% | 5.10 | 28.36 | 53.87 |

**Street scenario**

| Policy | Saved Episodes Used | Pass Rate | Timeout Rate | Avg. Crash Count | Avg. Speed (km/h) | Avg. Goal Time (s) | Avg. Sim Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Expert | 10 | 100.0% | 0.0% | 0.00 | 33.64 | 48.40 | 72.02 |
| IDM | 3 | 100.0% | 0.0% | 0.00 | 28.45 | 6.53 | 84.80* |

`*` The saved IDM street files are limited and not all of them include `episode_sim_time_s`, so the reported simulated time is only from the file that contains that field.

These tables form the basis of the following detailed discussion.

### 6.2 Highway Scenario Results

The highway scenario produces the clearest policy separation in the project. Among the saved results, IDM is the strongest overall policy. It completes forty-nine out of fifty episodes successfully and has only one timeout. Its average crash count of `1.04` is the lowest among the successful high-completion highway policies, and its average speed of `24.52 km/h` indicates stable forward progress without excessive caution.

The pure Expert Policy is the weakest highway baseline by a large margin. The workbook records zero successful passes across ten episodes and a timeout rate of one hundred percent. Its average speed is only `3.75 km/h`, which is dramatically lower than every other highway policy. This confirms the practical observation already noted inside the workbook itself: the Expert controller, as tested in this project’s highway configuration, is unable to handle the scenario properly. The issue is not simply a slightly worse score. It is a systematic mismatch between the policy and the scenario.

The hybrid Expert+IDM controller improves the situation substantially compared with pure Expert. With fifty saved episodes, it records an `82.0%` pass rate, which is a large improvement over the zero percent pass rate of Expert alone. This shows that the fallback strategy is valuable. The controller is able to recover from some of the situations in which the pure Expert policy stalls or becomes blocked. However, the hybrid controller still underperforms plain IDM. Its timeout rate is higher at `18.0%`, its average crash count rises to `3.52`, and its average speed falls below IDM. This means that the fallback logic helps but does not fully solve the highway weaknesses of the Expert-first approach.

The highway TrajectoryIDM controller presents a more complicated picture. At first glance, a `90.0%` pass rate seems strong. However, the average crash count is `35.40`, which is by far the worst safety result in the highway table. This is not a small degradation. It is a severe safety problem. The controller often reaches the destination, but it does so while colliding too frequently to be considered suitable for this scenario. This is exactly why the project uses multiple evaluation criteria. If completion alone had been used, this controller might have looked competitive. Once crash count is included, the conclusion changes completely.

From these results, the highway discussion leads to a clear scenario-specific conclusion. In the tested highway environment, plain IDM is the best policy among the saved results. Expert Policy is not suitable. Expert+IDM is a partial improvement but still weaker than IDM. TrajectoryIDM, while capable of reaching the goal in many runs, is not acceptable because of extremely poor collision behavior.

### 6.3 Discussion of Highway Policy Behavior

The code structure helps explain the highway outcomes.

The highway map includes not only moving traffic but also accident-related obstacles and static traffic objects. This means the policy must react quickly and maintain safe behavior when its preferred path is obstructed. IDM is fundamentally reactive and speed-based, so it fits this kind of environment reasonably well. It may not be elegant in every maneuver, but it continues to make progress with relatively low crash frequency.

The pure Expert policy appears to struggle because the tested scenario creates cases where its decision-making does not recover once the vehicle is blocked or slowed. This interpretation is supported by the creation of `StuckAwareExpertPolicy` in the project itself. That custom policy was clearly introduced because the author observed that the base Expert policy was not robust enough in the highway environment. The hybrid results confirm that this diagnosis was correct, because adding IDM fallback does improve the pass rate dramatically.

The remaining gap between Expert+IDM and pure IDM suggests that switching late into fallback mode may still be weaker than operating in a scenario-compatible reactive mode from the beginning. In other words, IDM seems not only useful as a rescue policy, but better suited as the primary policy in this specific highway setup.

The highway trajectory result is also understandable in implementation terms. Following a planned path is not the same as handling high-speed obstacle interaction safely. The obstacle-aware extension adds lane-change and recovery logic, but the saved result file shows that the controller still collides heavily. This suggests that, in the tested configuration, the trajectory-following behavior and avoidance logic did not integrate well enough with the dynamic structure of the highway environment.

### 6.4 Intersection Scenario Results

The intersection scenario gives a different ranking from the highway scenario, but the leading policy remains IDM. In the saved workbook, IDM completes all fifty out of fifty episodes successfully, giving a `100.0%` pass rate with zero timeouts. Its average crash count is `2.82`, which is not the lowest among the three policies but is substantially lower than the trajectory controller. Its average speed is `26.11 km/h`, and its average goal time is `50.60 s`, making it the strongest overall policy in this scenario as well.

The Expert Policy performs more mixedly in the intersection environment than it does in the highway environment. Unlike the highway case, it does achieve successful runs here. It records a `38.0%` pass rate and a `14.0%` timeout rate across fifty saved episodes. Its average crash count is `1.47`, which is actually lower than IDM and much lower than TrajectoryIDM. However, the low average speed of `15.12 km/h` and the relatively poor completion rate show that this lower crash figure comes at the cost of weak progress and inconsistent route completion.

The intersection TrajectoryIDM controller achieves a `72.0%` pass rate with zero timeouts. It also has the highest average speed at `28.36 km/h`. This means the controller is very effective at maintaining movement through the route. However, it also records the highest average crash count at `5.10`, which again shows a safety trade-off. In this scenario the trade-off is not as severe as the highway trajectory case, but it is still significant.

These results show that the intersection scenario is more balanced than the highway scenario. Expert is not a complete failure here, and TrajectoryIDM is more defensible than it is on the highway. Even so, IDM still provides the strongest overall combination of completion, stability, and acceptable safety.

### 6.5 Discussion of Intersection Policy Behavior

The intersection scenario differs from the highway case because the route is shorter, more structured, and more dependent on controlled navigation through conflict points. The target vehicle has a fixed destination through a custom map, and the surrounding traffic acts as interruption vehicles rather than as broad highway flow. This change appears to help Expert Policy somewhat, since it no longer collapses as completely as it did on the highway.

However, the Expert results also show that being cautious or low-crash is not enough if the policy does not complete the route reliably. The lower average speed and low pass rate suggest that Expert behavior in this environment is conservative to the point of weak task completion.

TrajectoryIDM performs better here than on the highway because the route structure aligns more naturally with explicit path following. The scenario has a clear geometric path from entry road to roundabout exit, so building a `PointLane` and following it is more appropriate than in a highway full of accident-related obstacles. Still, the increased crash count indicates that route adherence and movement efficiency can lead to more aggressive interaction with interruption traffic.

IDM again stands out because it balances progress and safety. It is not the fastest and not the lowest in crash count, but it completes all episodes and avoids timeouts entirely. In a scenario intended for reliable route negotiation, that balance is highly valuable.

### 6.6 Street Scenario Results

The street scenario result interpretation must remain strictly aligned with the saved files available in the repository. The strongest saved street result set is the ten-episode Expert workbook. In that file, Expert Policy completes all ten episodes successfully, with zero timeouts and zero crashes. The average speed is `33.64 km/h`, the average recorded goal time is `48.40 s`, and the average simulated episode time is `72.02 s`. These are very strong results and show that Expert Policy performs very well in the tested street configuration.

The saved IDM street outputs are smaller in number, but the available rows also show successful behavior. Across the saved IDM rows, pass rate is `100.0%`, timeout rate is `0.0%`, crash count is `0.00`, and average speed is `28.45 km/h`. Because the saved IDM street outputs are not as complete as the ten-episode Expert file, they should be interpreted more cautiously. However, the available evidence still shows that IDM can also handle the implemented street scenario successfully.

The street scenario is therefore different from both the highway and the intersection scenarios. It is the environment in which the saved Expert results are strongest. This is important because it shows again that policy quality depends on scenario match rather than on a universal ranking.

### 6.7 Discussion of Street Policy Behavior

The implementation of the street scenario helps explain why Expert performs strongly there. The route is narrower, the layout is deterministic, and the mid-route pedestrian controller is triggered in a structured way based on route progress and vehicle proximity. Although pedestrians introduce complexity, the overall road environment is not as obstacle-heavy as the highway scenario and not as multi-agent conflict-driven as the intersection scenario.

The street scripts also show careful engineering in how pedestrians are handled. They spawn at the curb, wait before crossing, move toward a defined target, and are cleaned up before resets. This avoids environment instability and makes the interaction pattern more controlled. Such structured event design may be more compatible with the Expert controller than the less predictable obstacle situations in the highway environment.

The available IDM street results also show clean performance, but because fewer saved rows are available, the report should not overstate direct superiority or infer a full ten-episode comparison that has not been saved in the current analysis folder. The correct conclusion is that both policies perform well in the recorded street results, with the most complete saved evidence belonging to the ten-episode Expert run.

### 6.8 Cross-Scenario Comparison

A cross-scenario comparison produces three major findings.

The first finding is that IDM is the most reliable all-round policy in the project. It performs best overall in the highway and intersection environments and also shows clean behavior in the saved street outputs. It may not always be the fastest policy and may not always have the absolute lowest crash count in every scenario, but it is the strongest overall baseline when completion and stability are prioritized together.

The second finding is that Expert Policy is highly scenario-sensitive. It fails completely in the tested highway setting, performs inconsistently in the intersection setting, and performs strongly in the street setting. This is one of the clearest proofs in the project that a policy cannot be judged outside the context of its environment.

The third finding is that trajectory-based control is beneficial only when its route-following advantages outweigh its interaction costs. In the intersection scenario, TrajectoryIDM is reasonably competitive because the route is fixed and geometrically meaningful. In the highway scenario, however, the same general style of control becomes unsuitable because the crash count becomes excessively high.

The project therefore supports a practical principle for autonomous driving evaluation: the best policy is not the one with the strongest name or the highest theoretical sophistication, but the one whose behavior matches the demands of the scenario while maintaining acceptable safety and completion performance.

### 6.9 Main Findings of the Project

Based on the implemented code and saved result files, the main findings are:

1. Plain IDM is the most reliable policy across the project’s recorded highway and intersection experiments.
2. Pure Expert Policy is not suitable for the tested highway configuration.
3. Adding IDM fallback improves Expert performance on the highway, but not enough to surpass plain IDM.
4. TrajectoryIDM can produce competitive movement and completion in route-structured scenarios, but it can also become unsafe if the scenario requires stronger reactive obstacle handling.
5. The street environment is the strongest saved scenario for Expert Policy, with the complete ten-episode Expert result file showing perfect completion and zero crashes.
6. The project’s custom scenario logic, especially in the intersection and street implementations, is sufficient to expose meaningful differences between policies.

## Chapter 7 Conclusion

This project developed a complete autonomous driving simulation and evaluation framework in MetaDrive based entirely on implemented project work. The project included three scenario categories: a highway environment with traffic and accident-related obstacles, a custom multi-agent intersection-roundabout environment with fixed target routing, and a street environment with dynamic pedestrian crossing behavior. Multiple policies were implemented and tested, including Expert Policy, IDM Policy, a hybrid Expert+IDM controller, and trajectory-based control using TrajectoryIDM.

The project results show clearly that autonomous driving policy performance is strongly dependent on scenario type. IDM produced the strongest overall performance in the highway and intersection result sets. It combined high completion with relatively controlled crash frequency and stable movement. Pure Expert Policy, despite being a strong built-in concept, was not suitable for the tested highway scenario and timed out in all saved highway episodes. The hybrid Expert+IDM controller improved on this and demonstrated that fallback design can recover some performance, but it still did not outperform plain IDM in the highway environment. TrajectoryIDM showed that route-guided control can work well in some route-structured conditions, especially the intersection scenario, but it also demonstrated that high completion without safety is not sufficient, as seen in the highway crash counts.

The street scenario provided an important contrast. In the saved ten-episode Expert workbook, Expert Policy completed every episode successfully without crashes or timeouts. The available IDM street results also showed successful runs without recorded crashes. This confirms that the street implementation in the project creates a scenario in which both tested policies can operate effectively, while still including meaningful pedestrian interaction.

The overall conclusion of the project is that there is no universal best policy across all tested environments. Instead, the correct policy choice depends on the relationship between the control strategy and the scenario structure. For the project’s highway and intersection work, IDM is the strongest overall choice. For the project’s street work, the saved Expert results are especially strong. The custom hybrid and trajectory-based controllers are valuable because they reveal both the potential and the limitations of adding recovery logic or explicit route following.

Most importantly, the project succeeds in its central goal: it creates a practical, scenario-based comparison framework in MetaDrive and uses that framework to generate clear, evidence-based conclusions from the project’s own saved outputs. This makes the work suitable as a final project report grounded directly in implementation and experiment rather than in unsupported general claims.
