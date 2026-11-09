"""Перелёт Luna-16 с низкой орбиты Кербина на орбиту Муны.

Скрипт предназначен для запуска после завершения ascent_autopilot.py. Он:

1. проверяет исходную орбиту и двигатель Poodle;
2. рассчитывает окно и уточняет манёвр по фактическому положению Муны;
3. выполняет разгонный импульс;
4. раскрывает дальние антенны и подтверждает солнечную зарядку;
5. ждёт входа в сферу влияния Муны, контролируя аккумулятор;
6. корректирует фактический перицентр после длинного разгонного импульса;
7. тормозит в перицентре на круговую орбиту высотой 50 км.

Перед выполнением манёвров требуется ввести ЛУНА. Скрипт автоматически делает
быстрое сохранение игры.
"""

from __future__ import annotations

import math
import time

import krpc


TARGET_MUN_ORBIT_ALTITUDE = 50_000.0
MIN_KERBIN_ORBIT_ALTITUDE = 75_000.0
MIN_SAFE_MUN_PERIAPSIS = 12_000.0
MUN_PERIAPSIS_TOLERANCE = 5_000.0
CORRECTION_LEAD_TIME = 180.0
MANEUVER_LEAD_TIME = 180.0
CAPTURE_RESERVE_DELTA_V = 350.0
LANDING_RESERVE_DELTA_V = 550.0
FUEL_UNIT_MASS_KG = 5.0
MIN_CONTROL_ELECTRIC_CHARGE = 25.0
MIN_SOLAR_FLOW = 0.01
SUNLIGHT_CLEARANCE = 20_000.0
SUNLIGHT_SEARCH_LIMIT = 2_400.0
ALIGNMENT_TIMEOUT = 60.0
SOLAR_TEST_DURATION = 8.0
SOI_MONITOR_STEP = 5_400.0
MIDCOURSE_TOLERANCE = 10_000.0
G0 = 9.80665


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def vector_subtract(left, right) -> tuple[float, float, float]:
    return tuple(a - b for a, b in zip(left, right))


def vector_dot(left, right) -> float:
    return sum(a * b for a, b in zip(left, right))


def vector_cross(left, right) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def vector_length(vector) -> float:
    return math.sqrt(vector_dot(vector, vector))


def normalized(vector) -> tuple[float, float, float]:
    length = vector_length(vector)
    if length <= 0.0:
        raise RuntimeError("Невозможно определить направление в пространстве.")
    return tuple(component / length for component in vector)


def engines_with_name(vessel, name_fragment: str):
    fragment = name_fragment.lower()
    return [
        engine
        for engine in vessel.parts.engines
        if fragment in engine.part.name.lower()
    ]


def available_fuel(vessel) -> tuple[float, float]:
    resources = vessel.resources
    return resources.amount("LiquidFuel"), resources.amount("Oxidizer")


def upper_stage_fuel(vessel) -> tuple[float, float]:
    """Return only propellant feeding Poodle, excluding lander reserves."""
    liquid_fuel = 0.0
    oxidizer = 0.0
    tank_names = {"Rockomax8BW", "Rockomax16.BW", "fuelTankSmallFlat"}
    for part in vessel.parts.all:
        if part.name not in tank_names:
            continue
        if part.resources.has_resource("LiquidFuel"):
            liquid_fuel += part.resources.amount("LiquidFuel")
        if part.resources.has_resource("Oxidizer"):
            oxidizer += part.resources.amount("Oxidizer")
    return liquid_fuel, oxidizer


def upper_stage_delta_v(vessel) -> float:
    liquid_fuel, oxidizer = upper_stage_fuel(vessel)
    propellant_mass = (liquid_fuel + oxidizer) * FUEL_UNIT_MASS_KG
    dry_mass = vessel.mass - propellant_mass
    isp = vessel.specific_impulse
    if propellant_mass <= 0.0 or dry_mass <= 0.0 or isp <= 0.0:
        return 0.0
    return isp * G0 * math.log(vessel.mass / dry_mass)


def estimate_burn_time(vessel, delta_v: float) -> float:
    thrust = max(vessel.available_thrust, vessel.max_thrust)
    isp = vessel.specific_impulse
    if thrust <= 0.0 or isp <= 0.0:
        raise RuntimeError("Нет доступной тяги для расчёта импульса.")

    exhaust_velocity = isp * G0
    initial_mass = vessel.mass
    final_mass = initial_mass / math.exp(delta_v / exhaust_velocity)
    mass_flow = thrust / exhaust_velocity
    return (initial_mass - final_mass) / mass_flow


def ensure_poodle(vessel) -> None:
    engines = engines_with_name(vessel, "liquidEngine2-2")
    if len(engines) != 1:
        raise RuntimeError(
            f"Ожидался один Poodle, найдено двигателей: {len(engines)}."
        )
    engine = engines[0]
    if not engine.has_fuel:
        raise RuntimeError("Poodle не имеет доступного топлива.")
    if not engine.active:
        engine.active = True


def electric_charge(vessel) -> tuple[float, float]:
    resources = vessel.resources
    return (
        resources.amount("ElectricCharge"),
        resources.max("ElectricCharge"),
    )


def solar_flow(vessel) -> float:
    return sum(panel.energy_flow for panel in vessel.parts.solar_panels)


def solar_exposure(vessel) -> float:
    panels = vessel.parts.solar_panels
    if not panels:
        return 0.0
    return max(panel.sun_exposure for panel in panels)


def deploy_long_range_antennas(vessel) -> None:
    """Раскрывает мощные тарелки до ухода от Кербина."""
    dishes = [
        antenna
        for antenna in vessel.parts.antennas
        if antenna.part.name == "commDish"
    ]
    if not dishes:
        raise RuntimeError("На корабле не найдены дальние антенны commDish.")

    for antenna in dishes:
        if antenna.deployable and not antenna.deployed:
            antenna.deployed = True

    deadline = time.monotonic() + 12.0
    while time.monotonic() < deadline:
        if all(not antenna.deployable or antenna.deployed for antenna in dishes):
            print(f"Дальние антенны раскрыты: {len(dishes)} шт.")
            return
        time.sleep(0.1)
    raise RuntimeError("Дальние антенны не успели раскрыться.")


def sunlight_clearance(conn, vessel) -> float:
    """Возвращает запас линии на Солнце над краем центрального тела."""
    body = vessel.orbit.body
    frame = body.non_rotating_reference_frame
    vessel_position = vessel.position(frame)
    sun_position = conn.space_center.bodies["Sun"].position(frame)
    ray = vector_subtract(sun_position, vessel_position)
    ray_length_squared = vector_dot(ray, ray)
    if ray_length_squared <= 0.0:
        return -body.equatorial_radius

    projection = -vector_dot(vessel_position, ray) / ray_length_squared
    if projection <= 0.0 or projection >= 1.0:
        return float("inf")
    closest_point = tuple(
        vessel_position[index] + projection * ray[index]
        for index in range(3)
    )
    return vector_length(closest_point) - body.equatorial_radius


def vessel_is_in_sunlight(conn, vessel) -> bool:
    """Проверяет, вышел ли корабль из тени с безопасным запасом."""
    return sunlight_clearance(conn, vessel) >= SUNLIGHT_CLEARANCE


def wait_for_sunlight(conn, vessel) -> None:
    """Безопасно выводит корабль из тени перед длинным перелётом."""
    if vessel_is_in_sunlight(conn, vessel):
        return

    start_ut = conn.space_center.ut
    print("Корабль находится в тени. Жду выхода на солнечный свет...")
    while not vessel_is_in_sunlight(conn, vessel):
        elapsed = conn.space_center.ut - start_ut
        if elapsed >= SUNLIGHT_SEARCH_LIMIT:
            raise RuntimeError(
                "Корабль не вышел из тени за 40 минут игрового времени."
            )
        charge, _ = electric_charge(vessel)
        if charge < MIN_CONTROL_ELECTRIC_CHARGE:
            raise RuntimeError(
                "Заряд стал критическим до выхода корабля на солнечный свет."
            )
        time_to_soi = vessel.orbit.time_to_soi_change
        if math.isfinite(time_to_soi) and time_to_soi <= 600.0:
            raise RuntimeError(
                "Слишком поздно искать Солнце перед входом в сферу Муны."
            )
        conn.space_center.warp_to(conn.space_center.ut + 120.0)
        time.sleep(0.2)

    print(
        f"Выход из тени подтверждён; луч проходит на "
        f"{sunlight_clearance(conn, vessel) / 1000.0:.1f} км выше края тела."
    )


def wait_for_alignment(
    vessel,
    autopilot,
    timeout: float = ALIGNMENT_TIMEOUT,
    require_roll: bool = False,
) -> None:
    """Ждёт наведения с ограничением времени и контролем аккумулятора."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        charge, _ = electric_charge(vessel)
        if charge < MIN_CONTROL_ELECTRIC_CHARGE:
            raise RuntimeError("Недостаточно электричества для ориентации корабля.")
        direction_ready = autopilot.error <= 3.0
        roll_ready = not require_roll or abs(autopilot.roll_error) <= 5.0
        if direction_ready and roll_ready:
            return
        time.sleep(0.1)
    raise RuntimeError("Автопилот не успел сориентировать корабль за 60 секунд.")


def orient_solar_panels(conn, vessel) -> None:
    """Перебирает безопасные положения и подтверждает реальную зарядку."""
    panels = vessel.parts.solar_panels
    if not panels:
        raise RuntimeError("На корабле нет солнечных панелей.")
    if not vessel_is_in_sunlight(conn, vessel):
        raise RuntimeError("Нельзя наводить панели, пока корабль находится в тени.")

    frame = vessel.orbit.body.non_rotating_reference_frame
    sun_position = conn.space_center.bodies["Sun"].position(frame)
    vessel_position = vessel.position(frame)
    sun_direction = normalized(vector_subtract(sun_position, vessel_position))
    helper_axis = (0.0, 0.0, 1.0)
    if abs(vector_dot(sun_direction, helper_axis)) > 0.9:
        helper_axis = (0.0, 1.0, 0.0)
    forward_direction = normalized(vector_cross(sun_direction, helper_axis))

    control = vessel.control
    autopilot = vessel.auto_pilot
    autopilot.reference_frame = frame

    # На V3 две OX-STAT установлены на нижней плоскости малого бака, поэтому
    # их рабочая нормаль близка к продольной оси корабля. Сначала проверяем оба
    # осевых положения, а затем четыре боковых на случай изменения конструкции.
    candidates = [
        (tuple(-component for component in sun_direction), 0.0, "хвостом"),
        (sun_direction, 0.0, "носом"),
    ]
    candidates.extend(
        (forward_direction, roll, f"боком, крен {roll:.0f}°")
        for roll in (0.0, 90.0, 180.0, 270.0)
    )

    best_candidate = None
    lowest_drain = float("inf")
    for direction, target_roll, description in candidates:
        autopilot.target_direction = direction
        autopilot.target_roll = target_roll
        autopilot.engaged = True
        wait_for_alignment(vessel, autopilot, require_roll=True)
        # Реакционные колёса во время наведения могут потреблять больше энергии,
        # чем дают две маленькие OX-STAT. Для проверки генерации оставляем
        # найденную ориентацию, но полностью отключаем наведение и SAS.
        autopilot.engaged = False
        control.sas = False
        time.sleep(1.0)
        flow = solar_flow(vessel)
        exposure = solar_exposure(vessel)
        if flow >= MIN_SOLAR_FLOW or exposure > 0.05:
            charge_before, maximum = electric_charge(vessel)
            time.sleep(SOLAR_TEST_DURATION)
            charge_after, _ = electric_charge(vessel)
            drain = max(
                0.0,
                (charge_before - charge_after) / SOLAR_TEST_DURATION,
            )
            if drain < lowest_drain:
                lowest_drain = drain
                best_candidate = (direction, target_roll, description)
            # На полном аккумуляторе KSP иногда возвращает energy_flow=0,
            # хотя панели освещены и покрывают текущую нагрузку. Поэтому
            # окончательное решение принимаем по реальному изменению заряда.
            if drain > 0.03:
                continue
            print(
                f"Солнечная зарядка подтверждена ({description} к Солнцу): "
                f"освещённость {exposure:.2f}, поток {flow:.3f} EC/с; "
                f"аккумулятор {charge_after:.1f}/{maximum:.1f}."
            )
            return

    autopilot.engaged = False
    control.sas = False
    exposure = solar_exposure(vessel)
    if exposure <= 0.0:
        raise RuntimeError(
            "Панели не видят Солнце ни в одном положении. "
            "Вероятно, их рабочая сторона закрыта соседними деталями."
        )

    # Даже при слабой генерации аппарат может безопасно долететь на запасе
    # аккумулятора. Оставляем его в положении с минимальным измеренным расходом
    # только если расчётного заряда хватает до сферы влияния Муны с запасом.
    if best_candidate is not None and math.isfinite(lowest_drain):
        direction, target_roll, description = best_candidate
        autopilot.target_direction = direction
        autopilot.target_roll = target_roll
        autopilot.engaged = True
        wait_for_alignment(vessel, autopilot, require_roll=True)
        autopilot.engaged = False
        control.sas = False
        charge, maximum = electric_charge(vessel)
        time_to_soi = vessel.orbit.time_to_soi_change
        required_charge = lowest_drain * max(0.0, time_to_soi + 1_800.0)
        required_charge += MIN_CONTROL_ELECTRIC_CHARGE
        if charge >= required_charge:
            print(
                f"Предупреждение: чистая зарядка не подтверждена; оставляю "
                f"корабль {description} к Солнцу. Пассивный расход около "
                f"{lowest_drain:.3f} EC/с, заряд {charge:.1f}/{maximum:.1f} "
                f"достаточен до Муны по текущей оценке."
            )
            return
    raise RuntimeError(
        "Панели освещены, но пассивный расход превышает генерацию, а заряда "
        "недостаточно до Муны. Нужно переставить или добавить солнечные панели."
    )


def wait_for_thrust(vessel, control, timeout: float = 4.0) -> None:
    ensure_poodle(vessel)
    control.throttle = 1.0
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if vessel.thrust >= 1.0:
            return
        time.sleep(0.05)
    control.throttle = 0.0
    raise RuntimeError("Poodle не создал тягу за четыре секунды.")


def execute_node(conn, vessel, node, description: str) -> None:
    control = vessel.control
    autopilot = vessel.auto_pilot
    delta_v = abs(node.delta_v)
    burn_time = estimate_burn_time(vessel, delta_v)
    burn_start_ut = node.ut - burn_time / 2.0

    print(
        f"{description}: Δv={delta_v:.1f} м/с, "
        f"расчётное горение={burn_time:.1f} с."
    )
    autopilot.reference_frame = node.reference_frame
    autopilot.target_direction = (0.0, 1.0, 0.0)
    autopilot.engaged = True
    wait_for_alignment(vessel, autopilot)

    conn.space_center.warp_to(
        max(conn.space_center.ut, burn_start_ut - 12.0)
    )
    wait_for_alignment(vessel, autopilot)
    if conn.space_center.ut > burn_start_ut + 2.0:
        raise RuntimeError(
            f"Пропущено начало манёвра '{description}' из-за задержки наведения."
        )
    while conn.space_center.ut < burn_start_ut:
        charge, _ = electric_charge(vessel)
        if charge < MIN_CONTROL_ELECTRIC_CHARGE:
            raise RuntimeError(
                f"Недостаточно электричества перед манёвром '{description}'."
            )
        time.sleep(0.05)

    wait_for_thrust(vessel, control)
    previous_remaining = float("inf")
    increasing_samples = 0
    zero_thrust_since = None

    while True:
        vector = node.remaining_burn_vector(node.reference_frame)
        remaining = math.sqrt(sum(component**2 for component in vector))
        control.throttle = clamp(remaining / 20.0, 0.06, 1.0)

        charge, _ = electric_charge(vessel)
        if charge < MIN_CONTROL_ELECTRIC_CHARGE:
            control.throttle = 0.0
            raise RuntimeError(
                f"Заряд стал критическим во время манёвра '{description}'."
            )

        if vessel.thrust < 1.0:
            if zero_thrust_since is None:
                zero_thrust_since = time.monotonic()
            if time.monotonic() - zero_thrust_since >= 1.0:
                control.throttle = 0.0
                poodle = engines_with_name(vessel, "liquidEngine2-2")
                if poodle and not any(engine.has_fuel for engine in poodle):
                    raise RuntimeError(
                        f"Poodle выработал топливо; в узле осталось "
                        f"{remaining:.1f} м/с."
                    )
                raise RuntimeError("Во время манёвра пропала тяга Poodle.")
        else:
            zero_thrust_since = None

        if remaining < 0.35:
            break
        if remaining > previous_remaining + 0.12:
            increasing_samples += 1
        else:
            increasing_samples = 0
        if increasing_samples >= 4:
            break

        previous_remaining = remaining
        time.sleep(0.05)

    control.throttle = 0.0
    node.remove()
    autopilot.engaged = False
    control.sas = True
    print(f"{description} завершён.")


def orbital_longitude(orbit) -> float:
    return (
        orbit.longitude_of_ascending_node
        + orbit.argument_of_periapsis
        + orbit.true_anomaly
    ) % (2.0 * math.pi)


def initial_transfer_solution(conn, vessel, mun) -> tuple[float, float]:
    orbit = vessel.orbit
    mu = orbit.body.gravitational_parameter
    r1 = orbit.semi_major_axis
    r2 = mun.orbit.semi_major_axis
    transfer_axis = (r1 + r2) / 2.0
    transfer_time = math.pi * math.sqrt(transfer_axis**3 / mu)
    prograde_delta_v = math.sqrt(mu / r1) * (
        math.sqrt(2.0 * r2 / (r1 + r2)) - 1.0
    )

    desired_phase = math.pi - mun.orbit.mean_motion * transfer_time
    current_phase = (
        orbital_longitude(mun.orbit) - orbital_longitude(orbit)
    ) % (2.0 * math.pi)
    relative_rate = orbit.mean_motion - mun.orbit.mean_motion
    wait_time = (
        (current_phase - desired_phase) % (2.0 * math.pi)
    ) / relative_rate
    synodic_period = 2.0 * math.pi / relative_rate
    burn_ut = conn.space_center.ut + wait_time
    while burn_ut < conn.space_center.ut + MANEUVER_LEAD_TIME:
        burn_ut += synodic_period
    return burn_ut, prograde_delta_v


def closest_distance_for_node(vessel, mun, ut: float, delta_v: float) -> float:
    node = vessel.control.add_node(ut, prograde=delta_v)
    try:
        approach = node.orbit.next_closest_approach(mun.orbit)
        return approach.distance
    finally:
        node.remove()


def approach_altitude_after_correction(
    vessel,
    mun,
    ut: float,
    radial: float,
) -> float:
    """Возвращает высоту сближения после временной ранней коррекции."""
    node = vessel.control.add_node(ut, radial=radial)
    try:
        approach = node.orbit.next_closest_approach(mun.orbit)
        return approach.distance - mun.equatorial_radius
    finally:
        node.remove()


def plan_kerbin_midcourse_correction(conn, vessel, mun):
    """Исправляет фактическую траекторию вскоре после разгона к Муне.

    Длинный 60–70-секундный импульс не совпадает с идеальным мгновенным узлом.
    Маленькая коррекция далеко от Муны намного дешевле, чем исправление
    перицентра после входа в её сферу влияния.
    """
    if vessel.orbit.body.name != "Kerbin":
        return None

    approach = vessel.orbit.next_closest_approach(mun.orbit)
    current_altitude = approach.distance - mun.equatorial_radius
    if abs(current_altitude - TARGET_MUN_ORBIT_ALTITUDE) <= MIDCOURSE_TOLERANCE:
        print(
            f"Ранняя коррекция не требуется: прогноз пролёта "
            f"{current_altitude / 1000.0:.1f} км."
        )
        return None

    time_to_soi = vessel.orbit.time_to_soi_change
    if not math.isfinite(time_to_soi) or time_to_soi <= 900.0:
        print("Для ранней коррекции уже мало времени; исправлю перицентр у Муны.")
        return None

    correction_ut = conn.space_center.ut + min(600.0, time_to_soi * 0.10)
    best_radial = 0.0
    best_error = float("inf")

    for radial in range(-150, 151, 5):
        altitude = approach_altitude_after_correction(
            vessel, mun, correction_ut, float(radial)
        )
        error = abs(altitude - TARGET_MUN_ORBIT_ALTITUDE)
        if error < best_error:
            best_error = error
            best_radial = float(radial)

    for step in (1.0, 0.1, 0.01):
        center = best_radial
        for offset in range(-5, 6):
            radial = center + offset * step
            altitude = approach_altitude_after_correction(
                vessel, mun, correction_ut, radial
            )
            error = abs(altitude - TARGET_MUN_ORBIT_ALTITUDE)
            if error < best_error:
                best_error = error
                best_radial = radial

    node = vessel.control.add_node(correction_ut, radial=best_radial)
    predicted = node.orbit.next_closest_approach(mun.orbit)
    predicted_altitude = predicted.distance - mun.equatorial_radius
    if predicted.distance >= mun.sphere_of_influence:
        node.remove()
        raise RuntimeError("Ранняя коррекция потеряла встречу с Муной.")

    print(
        f"Ранняя коррекция траектории: radial={best_radial:.2f} м/с; "
        f"прогноз пролёта {predicted_altitude / 1000.0:.1f} км."
    )
    return node


def plan_transfer_node(conn, vessel, mun):
    if vessel.control.nodes:
        nodes = list(vessel.control.nodes)
        if len(nodes) != 1:
            raise RuntimeError(
                "Перед запуском оставьте не более одного манёвренного узла."
            )
        node = nodes[0]
        if node.ut <= conn.space_center.ut + 60.0:
            raise RuntimeError("Существующий манёвренный узел уже слишком близко.")
        approach = node.orbit.next_closest_approach(mun.orbit)
        if approach.distance >= mun.sphere_of_influence:
            raise RuntimeError(
                "Существующий узел не формирует встречу с Муной. Удалите его."
            )
        print("Использую существующий узел перелёта из быстрого сохранения.")
        return node, approach.distance - mun.equatorial_radius

    base_ut, delta_v = initial_transfer_solution(conn, vessel, mun)
    target_distance = mun.equatorial_radius + TARGET_MUN_ORBIT_ALTITUDE

    # Сначала ищем подходящую сторону траектории крупным шагом, затем уточняем
    # время. Изменение времени на несколько секунд сдвигает точку встречи на
    # десятки километров, поэтому фиксированного фазового угла недостаточно.
    candidates: list[tuple[float, float]] = []
    for offset in range(-240, 241, 10):
        candidate_ut = base_ut + offset
        if candidate_ut < conn.space_center.ut + MANEUVER_LEAD_TIME:
            continue
        distance = closest_distance_for_node(
            vessel, mun, candidate_ut, delta_v
        )
        candidates.append((abs(distance - target_distance), candidate_ut))

    if not candidates:
        raise RuntimeError("Не удалось найти допустимое окно перелёта.")
    _, best_ut = min(candidates)

    fine_candidates: list[tuple[float, float, float]] = []
    for step in range(-40, 41):
        candidate_ut = best_ut + step * 0.25
        distance = closest_distance_for_node(
            vessel, mun, candidate_ut, delta_v
        )
        fine_candidates.append(
            (abs(distance - target_distance), candidate_ut, distance)
        )
    _, best_ut, predicted_distance = min(fine_candidates)

    if predicted_distance >= mun.sphere_of_influence:
        raise RuntimeError(
            "Расчётная траектория не входит в сферу влияния Муны."
        )

    node = vessel.control.add_node(best_ut, prograde=delta_v)
    return node, predicted_distance - mun.equatorial_radius


def enter_mun_soi(conn, vessel, mun) -> None:
    # Во время перелёта раз в полтора игровых часа проверяем аккумулятор и
    # состояние панелей. Последний короткий участок обрабатывается отдельно.
    for _ in range(100):
        if vessel.orbit.body.name == mun.name:
            return
        charge, maximum = electric_charge(vessel)
        flow = solar_flow(vessel)
        exposure = solar_exposure(vessel)
        if charge < MIN_CONTROL_ELECTRIC_CHARGE:
            raise RuntimeError("Заряд критический перед перелётом к Муне.")
        if (
            vessel_is_in_sunlight(conn, vessel)
            and flow < MIN_SOLAR_FLOW
            and exposure <= 0.05
        ):
            raise RuntimeError(
                "Корабль освещён, но панели не видят Солнце. Ускорение отменено."
            )
        time_to_change = vessel.orbit.time_to_soi_change
        if not math.isfinite(time_to_change) or time_to_change <= 0.0:
            raise RuntimeError("После разгона не найден вход в сферу влияния Муны.")
        print(
            f"До сферы влияния Муны: {time_to_change / 3600.0:.2f} ч.; "
            f"заряд {charge:.1f}/{maximum:.1f}, генерация {flow:.3f} EC/с, "
            f"освещённость {exposure:.2f}."
        )
        if time_to_change > SOI_MONITOR_STEP + 30.0:
            target_ut = conn.space_center.ut + SOI_MONITOR_STEP
        else:
            target_ut = conn.space_center.ut + time_to_change + 5.0
        conn.space_center.warp_to(target_ut)
        time.sleep(1.0)
    raise RuntimeError("Не удалось перейти в сферу влияния Муны.")


def periapsis_after_radial_burn(vessel, ut: float, radial: float) -> float:
    """Return predicted Mun periapsis for a temporary radial correction."""
    node = vessel.control.add_node(ut, radial=radial)
    try:
        return node.orbit.periapsis_altitude
    finally:
        node.remove()


def plan_mun_periapsis_correction(conn, vessel, mun):
    """Create a small correction that moves the Mun periapsis to 50 km.

    A long finite Kerbin-departure burn does not exactly match the impulsive
    maneuver-node prediction.  Once inside the Mun SOI the actual conic is
    known, so a short radial correction can target the periapsis reliably.
    """
    orbit = vessel.orbit
    if orbit.body.name != mun.name:
        raise RuntimeError("Коррекция перицентра возможна только у Муны.")

    error = orbit.periapsis_altitude - TARGET_MUN_ORBIT_ALTITUDE
    if (
        orbit.periapsis_altitude >= MIN_SAFE_MUN_PERIAPSIS
        and abs(error) <= MUN_PERIAPSIS_TOLERANCE
    ):
        return None

    time_to_periapsis = orbit.time_to_periapsis
    if not math.isfinite(time_to_periapsis) or time_to_periapsis < 90.0:
        raise RuntimeError(
            "Слишком поздно выполнять автоматическую коррекцию перицентра Муны."
        )

    lead_time = min(CORRECTION_LEAD_TIME, time_to_periapsis * 0.20)
    correction_ut = conn.space_center.ut + max(30.0, lead_time)

    # Radial-in/radial-out is very effective far before periapsis.  A coarse
    # scan handles either approach geometry; successive scans refine the burn.
    best_radial = 0.0
    best_error = float("inf")
    for radial in range(-250, 251, 10):
        periapsis = periapsis_after_radial_burn(
            vessel, correction_ut, float(radial)
        )
        candidate_error = abs(periapsis - TARGET_MUN_ORBIT_ALTITUDE)
        if candidate_error < best_error:
            best_error = candidate_error
            best_radial = float(radial)

    for step in (1.0, 0.1, 0.01):
        center = best_radial
        for offset in range(-10, 11):
            radial = center + offset * step
            periapsis = periapsis_after_radial_burn(
                vessel, correction_ut, radial
            )
            candidate_error = abs(periapsis - TARGET_MUN_ORBIT_ALTITUDE)
            if candidate_error < best_error:
                best_error = candidate_error
                best_radial = radial

    node = vessel.control.add_node(correction_ut, radial=best_radial)
    predicted_periapsis = node.orbit.periapsis_altitude
    if predicted_periapsis < MIN_SAFE_MUN_PERIAPSIS:
        node.remove()
        raise RuntimeError(
            "Не удалось рассчитать безопасную коррекцию перицентра Муны."
        )

    print(
        f"Коррекция перицентра: radial={best_radial:.2f} м/с; "
        f"прогноз Pe={predicted_periapsis / 1000.0:.1f} км."
    )
    return node


def plan_capture_node(conn, vessel, mun):
    orbit = vessel.orbit
    if orbit.body.name != mun.name:
        raise RuntimeError("Корабль ещё не находится в сфере влияния Муны.")
    if orbit.periapsis_altitude < MIN_SAFE_MUN_PERIAPSIS:
        raise RuntimeError(
            f"Опасный перицентр: {orbit.periapsis_altitude / 1000.0:.1f} км. "
            "Автоматический захват остановлен до столкновения."
        )

    mu = mun.gravitational_parameter
    periapsis_radius = mun.equatorial_radius + orbit.periapsis_altitude
    velocity_at_periapsis = math.sqrt(
        mu * (2.0 / periapsis_radius - 1.0 / orbit.semi_major_axis)
    )
    circular_velocity = math.sqrt(mu / periapsis_radius)
    capture_delta_v = velocity_at_periapsis - circular_velocity
    if capture_delta_v <= 0.0:
        raise RuntimeError("Не требуется тормозной импульс у Муны.")

    node_ut = conn.space_center.ut + orbit.time_to_periapsis
    return vessel.control.add_node(node_ut, prograde=-capture_delta_v)


def emergency_stop() -> None:
    try:
        conn = krpc.connect(name="Luna-16 Mun transfer emergency stop")
        vessel = conn.space_center.active_vessel
        if vessel is not None:
            vessel.control.throttle = 0.0
            vessel.auto_pilot.engaged = False
            vessel.control.sas = True
        conn.close()
    except Exception:
        pass


def has_active_mun_encounter(vessel, mun) -> bool:
    """Проверяет, что корабль уже летит к сфере влияния Муны."""
    time_to_change = vessel.orbit.time_to_soi_change
    if not math.isfinite(time_to_change) or time_to_change <= 0.0:
        return False
    try:
        approach = vessel.orbit.next_closest_approach(mun.orbit)
    except RuntimeError:
        return False
    return approach.distance < mun.sphere_of_influence


def run() -> None:
    conn = krpc.connect(name="Luna-16 Mun transfer autopilot")
    space_center = conn.space_center
    vessel = space_center.active_vessel
    if vessel is None:
        conn.close()
        raise RuntimeError("Активный корабль не найден.")

    mun = space_center.bodies["Mun"]
    orbit = vessel.orbit
    starting_body = orbit.body.name
    if starting_body not in {"Kerbin", mun.name}:
        conn.close()
        raise RuntimeError("Стартовать нужно с орбиты Кербина или в сфере Муны.")
    if (
        starting_body == "Kerbin"
        and orbit.periapsis_altitude < MIN_KERBIN_ORBIT_ALTITUDE
    ):
        conn.close()
        raise RuntimeError(
            "Исходная орбита пересекает атмосферу Кербина. "
            "Сначала поднимите перицентр выше 75 км."
        )
    ensure_poodle(vessel)
    deploy_long_range_antennas(vessel)
    resuming_transfer = (
        starting_body == "Kerbin"
        and not vessel.control.nodes
        and has_active_mun_encounter(vessel, mun)
    )

    print(f"Корабль: {vessel.name}")
    if starting_body == "Kerbin":
        liquid_fuel, oxidizer = available_fuel(vessel)
        upper_liquid_fuel, upper_oxidizer = upper_stage_fuel(vessel)
        available_upper_delta_v = upper_stage_delta_v(vessel)
        print(
            f"Орбита Кербина: Pe={orbit.periapsis_altitude / 1000.0:.1f} км, "
            f"Ap={orbit.apoapsis_altitude / 1000.0:.1f} км."
        )
        print(
            f"Топливо: LiquidFuel={liquid_fuel:.1f}, "
            f"Oxidizer={oxidizer:.1f}."
        )
        print(
            f"В баках Poodle: LiquidFuel={upper_liquid_fuel:.1f}, "
            f"Oxidizer={upper_oxidizer:.1f}; "
            f"доступно около {available_upper_delta_v:.0f} м/с."
        )
        print(
            f"Цель: круговая орбита Муны около "
            f"{TARGET_MUN_ORBIT_ALTITUDE / 1000.0:.0f} км."
        )

        if resuming_transfer:
            print(
                "Обнаружена уже сформированная встреча с Муной. "
                "Повторный разгон не создаю; продолжаю перелёт."
            )
        else:
            transfer_node, predicted_periapsis = plan_transfer_node(
                conn, vessel, mun
            )
            required_delta_v = (
                transfer_node.delta_v
                + CAPTURE_RESERVE_DELTA_V
                + LANDING_RESERVE_DELTA_V
            )
            if available_upper_delta_v < required_delta_v:
                transfer_node.remove()
                conn.close()
                raise RuntimeError(
                    f"Топлива Poodle недостаточно: доступно около "
                    f"{available_upper_delta_v:.0f} м/с, требуется не менее "
                    f"{required_delta_v:.0f} м/с с резервом на захват Муны "
                    f"({CAPTURE_RESERVE_DELTA_V:.0f} м/с) и помощь при посадке "
                    f"({LANDING_RESERVE_DELTA_V:.0f} м/с). Добавьте бак "
                    "X200-8 к ступени Poodle."
                )
            wait_time = transfer_node.ut - space_center.ut
            print(
                f"Разгон через {wait_time / 60.0:.1f} мин; "
                f"Δv={transfer_node.delta_v:.1f} м/с."
            )
            print(
                f"Расчётная высота пролёта над Муной: "
                f"{predicted_periapsis / 1000.0:.1f} км."
            )
            if input("Для начала перелёта введите ЛУНА: ").strip().upper() != "ЛУНА":
                transfer_node.remove()
                conn.close()
                print("Перелёт отменён; расчётный узел удалён.")
                return

            if space_center.can_save:
                space_center.quicksave()
                print("Создано быстрое сохранение перед перелётом.")

            execute_node(conn, vessel, transfer_node, "Разгон к Муне")
            time_to_change = vessel.orbit.time_to_soi_change
            if not math.isfinite(time_to_change) or time_to_change <= 0.0:
                raise RuntimeError("Разгон не сформировал встречу с Муной.")
        midcourse_node = plan_kerbin_midcourse_correction(
            conn, vessel, mun
        )
        if midcourse_node is not None:
            execute_node(
                conn,
                vessel,
                midcourse_node,
                "Ранняя коррекция траектории к Муне",
            )
            approach = vessel.orbit.next_closest_approach(mun.orbit)
            print(
                f"Прогноз после ранней коррекции: "
                f"{(approach.distance - mun.equatorial_radius) / 1000.0:.1f} км."
            )
        wait_for_sunlight(conn, vessel)
        orient_solar_panels(conn, vessel)
        enter_mun_soi(conn, vessel, mun)
    elif orbit.semi_major_axis > 0.0 and orbit.periapsis_altitude >= 0.0:
        print(
            f"Корабль уже находится на связанной орбите Муны: "
            f"Pe={orbit.periapsis_altitude / 1000.0:.1f} км, "
            f"Ap={orbit.apoapsis_altitude / 1000.0:.1f} км."
        )
        conn.close()
        return
    else:
        print("Продолжаю прерванный подлёт внутри сферы влияния Муны.")

    print(
        f"Вход в сферу влияния Муны. Перицентр: "
        f"{vessel.orbit.periapsis_altitude / 1000.0:.1f} км."
    )
    correction_node = plan_mun_periapsis_correction(
        conn, vessel, mun
    )
    if correction_node is not None:
        execute_node(conn, vessel, correction_node, "Коррекция перицентра Муны")
        print(
            f"Перицентр после коррекции: "
            f"{vessel.orbit.periapsis_altitude / 1000.0:.1f} км."
        )

    capture_node = plan_capture_node(conn, vessel, mun)
    execute_node(conn, vessel, capture_node, "Торможение у Муны")

    orbit = vessel.orbit
    liquid_fuel, oxidizer = available_fuel(vessel)
    print(
        f"Орбита Муны достигнута: Pe={orbit.periapsis_altitude / 1000.0:.1f} км, "
        f"Ap={orbit.apoapsis_altitude / 1000.0:.1f} км."
    )
    print(
        f"Остаток топлива: LiquidFuel={liquid_fuel:.1f}, "
        f"Oxidizer={oxidizer:.1f}."
    )
    conn.close()


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        emergency_stop()
        print("\nПерелёт остановлен пользователем.")
    except Exception as error:
        emergency_stop()
        print(f"\nАВАРИЙНАЯ ОСТАНОВКА: {error}")
        raise
