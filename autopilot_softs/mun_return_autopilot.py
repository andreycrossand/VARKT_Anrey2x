"""Автоматическое возвращение Luna-16 с поверхности Муны на Кербин.

Скрипт выполняет всю обратную часть миссии одним запуском:

1. проверяет возвращаемую ступень, капсулу и запас delta-v;
2. адресно отделяет взлётный зонд от посадочной платформы;
3. выводит аппарат на низкую орбиту Муны;
4. рассчитывает и выполняет перелёт к Кербину;
5. уточняет перицентр входа в атмосферу;
6. отделяет возвращаемую капсулу;
7. удерживает теплозащиту по потоку и раскрывает парашют;
8. контролирует посадку или приводнение.

Перед запуском аппарат должен неподвижно стоять на Муне. Для фактического
старта требуется ввести ДОМОЙ. Скрипт делает быстрое сохранение до отделения.
"""

from __future__ import annotations

import math
import time

import krpc


EXPECTED_NAME_FRAGMENT = "Luna-16 Mun Final"

HEADING = 90.0
TARGET_MUN_ORBIT_ALTITUDE = 18_000.0
MIN_SAFE_MUN_ORBIT_ALTITUDE = 12_000.0
ASCENT_APOAPSIS_CUTOFF_MARGIN = 500.0
ASCENT_TIMEOUT = 480.0
TARGET_KERBIN_PERIAPSIS = 32_000.0
MIN_SAFE_KERBIN_PERIAPSIS = 20_000.0
MAX_SAFE_KERBIN_PERIAPSIS = 45_000.0

MIN_ELECTRIC_CHARGE = 10.0
MIN_START_DELTA_V = 1_550.0
TRANSFER_RESERVE_DELTA_V = 180.0
MANEUVER_LEAD_TIME = 300.0
ALIGNMENT_TIMEOUT = 90.0
ENGINE_START_TIMEOUT = 4.0
CAPSULE_SEPARATION_ALTITUDE = 200_000.0
ATMOSPHERE_APPROACH_ALTITUDE = 78_000.0
PARACHUTE_DEPLOY_ALTITUDE = 8_000.0
PARACHUTE_ARMED_DEPLOY_ALTITUDE = 5_000.0
PARACHUTE_DEPLOY_SPEED = 300.0
G0 = 9.80665

RETURN_ENGINE_NAMES = {"microEngine.v2", "radialEngineMini.v2"}


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def lerp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    if x1 == x0:
        return y1
    fraction = clamp((x - x0) / (x1 - x0), 0.0, 1.0)
    return y0 + fraction * (y1 - y0)


def vector_scale(vector, factor: float):
    return tuple(component * factor for component in vector)


def vector_dot(left, right) -> float:
    return sum(a * b for a, b in zip(left, right))


def vector_length(vector) -> float:
    return math.sqrt(vector_dot(vector, vector))


def normalized(vector):
    length = vector_length(vector)
    if length <= 0.0:
        raise RuntimeError("Не удалось определить направление аппарата.")
    return tuple(component / length for component in vector)


def engines_with_name(vessel, name: str):
    return [engine for engine in vessel.parts.engines if engine.part.name == name]


def return_engines(vessel):
    return [
        engine
        for engine in vessel.parts.engines
        if engine.part.name in RETURN_ENGINE_NAMES
        and engine.part.decouple_stage < 0
    ]


def descendants(part):
    result = []
    pending = list(part.children)
    while pending:
        current = pending.pop()
        result.append(current)
        pending.extend(current.children)
    return result


def has_part(vessel, name: str) -> bool:
    return any(part.name == name for part in vessel.parts.all)


def electric_charge(vessel) -> tuple[float, float]:
    resources = vessel.resources
    return (
        resources.amount("ElectricCharge"),
        resources.max("ElectricCharge"),
    )


def weighted_engine_performance(engines) -> tuple[float, float]:
    thrust = sum(engine.max_vacuum_thrust for engine in engines)
    if thrust <= 0.0:
        return 0.0, 0.0
    impulse = sum(
        engine.max_vacuum_thrust * engine.vacuum_specific_impulse
        for engine in engines
    )
    return thrust, impulse / thrust


def available_delta_v(vessel, engines) -> float:
    thrust, specific_impulse = weighted_engine_performance(engines)
    if thrust <= 0.0 or specific_impulse <= 0.0:
        return 0.0
    resources = vessel.resources
    liquid_fuel = resources.amount("LiquidFuel")
    oxidizer = resources.amount("Oxidizer")
    propellant_mass = (liquid_fuel + oxidizer) * 5.0
    dry_mass = vessel.mass - propellant_mass
    if propellant_mass <= 0.0 or dry_mass <= 0.0:
        return 0.0
    return specific_impulse * G0 * math.log(vessel.mass / dry_mass)


def predicted_ascent_stage_budget(vessel) -> tuple[float, float, float]:
    """Возвращает массу, TWR и delta-v после отделения посадочной платформы."""
    separator_part = find_ascent_separator(vessel).part
    removed = {separator_part, *descendants(separator_part)}
    remaining_parts = [part for part in vessel.parts.all if part not in removed]
    engines = return_engines(vessel)
    thrust, specific_impulse = weighted_engine_performance(engines)
    mass = sum(part.mass for part in remaining_parts)
    liquid_fuel = sum(
        part.resources.amount("LiquidFuel")
        for part in remaining_parts
        if part.resources.has_resource("LiquidFuel")
    )
    oxidizer = sum(
        part.resources.amount("Oxidizer")
        for part in remaining_parts
        if part.resources.has_resource("Oxidizer")
    )
    propellant_mass = (liquid_fuel + oxidizer) * 5.0
    dry_mass = mass - propellant_mass
    if thrust <= 0.0 or specific_impulse <= 0.0 or dry_mass <= 0.0:
        return mass, 0.0, 0.0
    twr = thrust / max(mass * vessel.orbit.body.surface_gravity, 1.0)
    delta_v = specific_impulse * G0 * math.log(mass / dry_mass)
    return mass, twr, delta_v


def estimate_burn_time(vessel, engines, delta_v: float) -> float:
    thrust, specific_impulse = weighted_engine_performance(engines)
    if thrust <= 0.0 or specific_impulse <= 0.0:
        raise RuntimeError("Нет доступной тяги для расчёта импульса.")
    exhaust_velocity = specific_impulse * G0
    final_mass = vessel.mass / math.exp(delta_v / exhaust_velocity)
    mass_flow = thrust / exhaust_velocity
    return (vessel.mass - final_mass) / mass_flow


def find_ascent_separator(vessel):
    """Находит TD-06 между зондом и посадочной платформой."""
    candidates = []
    for part in vessel.parts.all:
        try:
            decoupler = part.decoupler
        except RuntimeError:
            continue
        if decoupler is None or part.name != "Decoupler.0":
            continue
        below = descendants(part)
        landing_engines = sum(
            child.name == "smallRadialEngine" for child in below
        )
        parent = part.parent
        if (
            landing_engines == 8
            and parent is not None
            and parent.name == "microEngine.v2"
        ):
            candidates.append(decoupler)
    if len(candidates) != 1:
        raise RuntimeError(
            "Не удалось однозначно найти разделитель взлётного зонда "
            f"(найдено: {len(candidates)})."
        )
    return candidates[0]


def find_capsule_separator(vessel):
    """Находит верхний TD-06, под которым находится капсула ОКТО2."""
    candidates = []
    for part in vessel.parts.all:
        try:
            decoupler = part.decoupler
        except RuntimeError:
            continue
        if decoupler is None or part.name != "Decoupler.0":
            continue
        below = descendants(part)
        names = {child.name for child in below}
        if "probeCoreOcto2.v2" in names and "parachuteSingle" in names:
            candidates.append(decoupler)
    if len(candidates) != 1:
        raise RuntimeError(
            "Не удалось однозначно найти отделитель возвращаемой капсулы "
            f"(найдено: {len(candidates)})."
        )
    return candidates[0]


def validate_landed_vehicle(vessel) -> None:
    body = vessel.orbit.body
    if body.name != "Mun":
        raise RuntimeError("Возвращение нужно начинать с поверхности Муны.")
    situation_name = str(vessel.situation).lower()
    if "landed" not in situation_name:
        raise RuntimeError("Аппарат должен неподвижно стоять на поверхности Муны.")
    if EXPECTED_NAME_FRAGMENT.lower() not in vessel.name.lower():
        print(
            f"Предупреждение: имя аппарата — '{vessel.name}'. "
            "Проверяю конструкцию по деталям."
        )
    if len(engines_with_name(vessel, "smallRadialEngine")) != 8:
        raise RuntimeError(
            "Посадочная платформа не соответствует финальной сборке."
        )
    engines = return_engines(vessel)
    if len(engines) != 5:
        raise RuntimeError(
            f"Ожидались один 'Муравей' и четыре 'Паука'; найдено: {len(engines)}."
        )
    if len(vessel.parts.parachutes) != 1:
        raise RuntimeError("Ожидался один парашют возвращаемой капсулы.")
    if not has_part(vessel, "probeCoreOcto2.v2"):
        raise RuntimeError("Командный блок возвращаемой капсулы ОКТО2 не найден.")
    find_ascent_separator(vessel)
    find_capsule_separator(vessel)


def wait_for_alignment(vessel, autopilot, timeout: float = ALIGNMENT_TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        charge, _ = electric_charge(vessel)
        if charge < MIN_ELECTRIC_CHARGE:
            raise RuntimeError("Заряд стал критическим во время наведения.")
        if autopilot.error <= 4.0:
            return
        time.sleep(0.1)
    raise RuntimeError(
        f"Аппарат не успел сориентироваться за {timeout:.0f} секунд."
    )


def wait_for_return_stage(space_center, timeout: float = 12.0):
    deadline = time.monotonic() + timeout
    stable_reads = 0
    while time.monotonic() < deadline:
        candidates = []
        active = space_center.active_vessel
        if active is not None:
            candidates.append(active)
        for candidate in space_center.vessels:
            if candidate not in candidates:
                candidates.append(candidate)
        for candidate in candidates:
            try:
                engines = return_engines(candidate)
                landing = engines_with_name(candidate, "smallRadialEngine")
                if len(engines) == 5 and not landing:
                    _ = candidate.mass
                    if candidate == active:
                        stable_reads += 1
                    else:
                        space_center.active_vessel = candidate
                        stable_reads = 1
                    if stable_reads >= 4:
                        return candidate
                    break
            except RuntimeError:
                continue
        else:
            stable_reads = 0
        time.sleep(0.1)
    raise RuntimeError("После отделения не удалось выбрать взлётный зонд.")


def wait_for_capsule(space_center, detached, timeout: float = 12.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        candidates = []
        if detached is not None:
            candidates.append(detached)
        candidates.extend(space_center.vessels)
        for candidate in candidates:
            try:
                names = {part.name for part in candidate.parts.all}
                if (
                    "probeCoreOcto2.v2" in names
                    and "parachuteSingle" in names
                    and not candidate.parts.engines
                ):
                    space_center.active_vessel = candidate
                    _ = candidate.mass
                    return candidate
            except RuntimeError:
                continue
        time.sleep(0.1)
    raise RuntimeError("После отделения не удалось выбрать возвращаемую капсулу.")


def activate_engines(vessel, engines) -> None:
    for engine in engines:
        if not engine.has_fuel:
            raise RuntimeError(
                f"Двигатель {engine.part.title} не имеет доступного топлива."
            )
        if not engine.active:
            engine.active = True
    vessel.control.throttle = 1.0
    deadline = time.monotonic() + ENGINE_START_TIMEOUT
    while time.monotonic() < deadline:
        if vessel.thrust >= 1.0:
            return
        time.sleep(0.05)
    vessel.control.throttle = 0.0
    raise RuntimeError("Возвращаемые двигатели не создали тягу.")


def commanded_pitch(altitude: float) -> float:
    schedule = (
        (0.0, 90.0),
        (250.0, 90.0),
        (800.0, 82.0),
        (1_800.0, 68.0),
        (3_500.0, 50.0),
        (6_000.0, 32.0),
        (9_000.0, 18.0),
        (13_000.0, 7.0),
        (17_000.0, 2.0),
    )
    for (h0, p0), (h1, p1) in zip(schedule, schedule[1:]):
        if altitude <= h1:
            return lerp(altitude, h0, h1, p0, p1)
    return schedule[-1][1]


def launch_from_mun(space_center, vessel):
    separator = find_ascent_separator(vessel)
    control = vessel.control
    autopilot = vessel.auto_pilot
    control.throttle = 0.0
    control.sas = False
    control.rcs = False
    autopilot.reference_frame = vessel.surface_reference_frame
    autopilot.target_pitch_and_heading(90.0, HEADING)
    autopilot.engaged = True

    print("Отделяю взлётный зонд от посадочной платформы.")
    separator.decouple()
    time.sleep(0.35)
    vessel = wait_for_return_stage(space_center)
    control = vessel.control
    autopilot = vessel.auto_pilot
    autopilot.reference_frame = vessel.surface_reference_frame
    autopilot.target_pitch_and_heading(90.0, HEADING)
    autopilot.engaged = True
    engines = return_engines(vessel)

    activate_engines(vessel, engines)
    print("Старт с Муны. Пять возвращаемых двигателей подтвердили тягу.")
    start_ut = space_center.ut
    last_report_altitude = -2_000.0
    target_apoapsis = TARGET_MUN_ORBIT_ALTITUDE

    while (
        vessel.orbit.apoapsis_altitude
        < target_apoapsis - ASCENT_APOAPSIS_CUTOFF_MARGIN
    ):
        flight = vessel.flight(vessel.orbit.body.reference_frame)
        altitude = max(0.0, flight.surface_altitude)
        pitch = commanded_pitch(altitude)
        autopilot.target_pitch_and_heading(pitch, HEADING)

        apoapsis = vessel.orbit.apoapsis_altitude
        throttle = 1.0
        if apoapsis > target_apoapsis - 1_000.0:
            throttle = lerp(
                apoapsis,
                target_apoapsis - 1_000.0,
                target_apoapsis,
                1.0,
                0.40,
            )
        control.throttle = throttle

        if vessel.thrust < 1.0:
            control.throttle = 0.0
            raise RuntimeError("Тяга пропала во время взлёта с Муны.")
        if space_center.ut - start_ut > ASCENT_TIMEOUT:
            control.throttle = 0.0
            if apoapsis >= MIN_SAFE_MUN_ORBIT_ALTITUDE:
                print(
                    f"Предупреждение: достигнут безопасный апоцентр "
                    f"{apoapsis / 1000.0:.1f} км; перехожу к циркуляризации."
                )
                break
            raise RuntimeError(
                "Взлёт не достиг безопасного апоцентра за 8 минут."
            )
        if altitude >= last_report_altitude + 2_000.0:
            print(
                f"Высота {altitude / 1000.0:.1f} км; "
                f"апоцентр {apoapsis / 1000.0:.1f} км; тангаж {pitch:.0f}°."
            )
            last_report_altitude = altitude
        time.sleep(0.05)

    control.throttle = 0.0
    print(
        f"Апоцентр достигнут: "
        f"{vessel.orbit.apoapsis_altitude / 1000.0:.1f} км."
    )
    return vessel, engines


def circularization_delta_v(vessel) -> float:
    orbit = vessel.orbit
    body = orbit.body
    apoapsis_radius = body.equatorial_radius + orbit.apoapsis_altitude
    velocity_at_apoapsis = math.sqrt(
        body.gravitational_parameter
        * (2.0 / apoapsis_radius - 1.0 / orbit.semi_major_axis)
    )
    circular_velocity = math.sqrt(
        body.gravitational_parameter / apoapsis_radius
    )
    return max(0.0, circular_velocity - velocity_at_apoapsis)


def execute_node(conn, vessel, engines, node, description: str) -> None:
    control = vessel.control
    autopilot = vessel.auto_pilot
    delta_v = abs(node.delta_v)
    burn_time = estimate_burn_time(vessel, engines, delta_v)
    burn_start_ut = node.ut - burn_time / 2.0
    print(
        f"{description}: delta-v={delta_v:.1f} м/с, "
        f"горение около {burn_time:.1f} с."
    )

    autopilot.reference_frame = node.reference_frame
    autopilot.target_direction = (0.0, 1.0, 0.0)
    control.sas = False
    autopilot.engaged = True
    wait_for_alignment(vessel, autopilot)

    conn.space_center.warp_to(
        max(conn.space_center.ut, burn_start_ut - 12.0)
    )
    wait_for_alignment(vessel, autopilot)
    while conn.space_center.ut < burn_start_ut:
        time.sleep(0.05)

    activate_engines(vessel, engines)
    previous_remaining = float("inf")
    increasing_samples = 0
    zero_thrust_since = None
    while True:
        vector = node.remaining_burn_vector(node.reference_frame)
        remaining = vector_length(vector)
        control.throttle = clamp(remaining / 18.0, 0.06, 1.0)

        if vessel.thrust < 1.0:
            if zero_thrust_since is None:
                zero_thrust_since = time.monotonic()
            elif time.monotonic() - zero_thrust_since >= 1.0:
                control.throttle = 0.0
                raise RuntimeError(
                    f"Во время манёвра '{description}' пропала тяга; "
                    f"осталось {remaining:.1f} м/с."
                )
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
    # Между манёврами аппарат летит пассивно. Постоянный SAS на несколько
    # игровых часов незаметно разряжал небольшой аккумулятор ступени.
    control.sas = False
    print(f"{description} завершён.")


def circularize_mun_orbit(conn, vessel, engines) -> None:
    delta_v = circularization_delta_v(vessel)
    if delta_v <= 0.0:
        raise RuntimeError("Не удалось рассчитать циркуляризацию у Муны.")
    node_ut = conn.space_center.ut + vessel.orbit.time_to_apoapsis
    node = vessel.control.add_node(node_ut, prograde=delta_v)
    execute_node(conn, vessel, engines, node, "Выход на орбиту Муны")
    periapsis = vessel.orbit.periapsis_altitude
    apoapsis = vessel.orbit.apoapsis_altitude
    if periapsis < MIN_SAFE_MUN_ORBIT_ALTITUDE:
        raise RuntimeError(
            f"Получен слишком низкий перицентр Муны: {periapsis / 1000.0:.1f} км."
        )
    print(
        f"Орбита Муны получена: Pe={periapsis / 1000.0:.1f} км, "
        f"Ap={apoapsis / 1000.0:.1f} км."
    )


def patched_orbit_for_body(orbit, body_name: str):
    current = orbit
    for _ in range(6):
        if current.body.name == body_name:
            return current
        try:
            current = current.next_orbit
        except RuntimeError:
            return None
        if current is None:
            return None
    return None


def kerbin_periapsis_for_node(vessel, ut: float, prograde: float):
    node = vessel.control.add_node(ut, prograde=prograde)
    try:
        kerbin_orbit = patched_orbit_for_body(node.orbit, "Kerbin")
        if kerbin_orbit is None:
            return None
        return kerbin_orbit.periapsis_altitude
    except RuntimeError:
        return None
    finally:
        node.remove()


def plan_kerbin_transfer(conn, vessel):
    """Численно ищет окно ухода, дающее атмосферный перицентр Кербина."""
    orbit = vessel.orbit
    if orbit.body.name != "Mun":
        raise RuntimeError("Уход к Кербину рассчитывается только на орбите Муны.")
    now = conn.space_center.ut
    period = orbit.period
    best = None

    for time_step in range(25):
        ut = now + MANEUVER_LEAD_TIME + period * time_step / 24.0
        for delta_v in range(220, 421, 20):
            periapsis = kerbin_periapsis_for_node(
                vessel, ut, float(delta_v)
            )
            if periapsis is None:
                continue
            candidate = (
                abs(periapsis - TARGET_KERBIN_PERIAPSIS),
                ut,
                float(delta_v),
                periapsis,
            )
            if best is None or candidate < best:
                best = candidate

    if best is None:
        raise RuntimeError("Не удалось найти траекторию с Муны к Кербину.")

    _, coarse_ut, coarse_delta_v, _ = best
    time_step = period / 24.0
    for offset_index in range(-12, 13):
        ut = coarse_ut + offset_index * time_step / 12.0
        if ut < now + 120.0:
            continue
        for delta_index in range(-8, 9):
            delta_v = coarse_delta_v + delta_index * 2.5
            periapsis = kerbin_periapsis_for_node(vessel, ut, delta_v)
            if periapsis is None:
                continue
            candidate = (
                abs(periapsis - TARGET_KERBIN_PERIAPSIS),
                ut,
                delta_v,
                periapsis,
            )
            if candidate < best:
                best = candidate

    error, best_ut, best_delta_v, predicted_periapsis = best
    if error > 25_000.0:
        raise RuntimeError(
            "Расчёт не смог получить безопасный перицентр Кербина: "
            f"{predicted_periapsis / 1000.0:.1f} км."
        )
    node = vessel.control.add_node(best_ut, prograde=best_delta_v)
    return node, predicted_periapsis


def wait_for_kerbin_soi(conn, vessel) -> None:
    print("Перелёт к Кербину. Контролирую вход в его сферу влияния.")
    for _ in range(12):
        if vessel.orbit.body.name == "Kerbin":
            return
        charge, maximum = electric_charge(vessel)
        if charge < MIN_ELECTRIC_CHARGE:
            raise RuntimeError("Заряд стал критическим во время перелёта.")
        time_to_change = vessel.orbit.time_to_soi_change
        if not math.isfinite(time_to_change) or time_to_change <= 0.0:
            raise RuntimeError("После разгона не найдено возвращение к Кербину.")
        print(
            f"До сферы Кербина {time_to_change / 3600.0:.2f} ч.; "
            f"заряд {charge:.1f}/{maximum:.1f} EC."
        )
        conn.space_center.warp_to(
            conn.space_center.ut + max(1.0, time_to_change + 3.0)
        )
        time.sleep(1.0)
    raise RuntimeError("Не удалось перейти в сферу влияния Кербина.")


def kerbin_periapsis_after_correction(
    vessel,
    ut: float,
    prograde: float,
    radial: float,
) -> float:
    node = vessel.control.add_node(ut, prograde=prograde, radial=radial)
    try:
        return node.orbit.periapsis_altitude
    finally:
        node.remove()


def plan_kerbin_correction(conn, vessel):
    orbit = vessel.orbit
    periapsis = orbit.periapsis_altitude
    if MIN_SAFE_KERBIN_PERIAPSIS <= periapsis <= MAX_SAFE_KERBIN_PERIAPSIS:
        print(
            f"Коррекция не требуется: перицентр Кербина "
            f"{periapsis / 1000.0:.1f} км."
        )
        return None

    time_to_periapsis = orbit.time_to_periapsis
    if not math.isfinite(time_to_periapsis) or time_to_periapsis < 600.0:
        raise RuntimeError("Слишком поздно исправлять траекторию входа.")
    correction_ut = conn.space_center.ut + min(600.0, time_to_periapsis * 0.10)
    best_prograde = 0.0
    best_radial = 0.0
    best_error = abs(periapsis - TARGET_KERBIN_PERIAPSIS)

    for prograde in range(-150, 151, 10):
        candidate_periapsis = kerbin_periapsis_after_correction(
            vessel, correction_ut, float(prograde), 0.0
        )
        error = abs(candidate_periapsis - TARGET_KERBIN_PERIAPSIS)
        if error < best_error:
            best_error = error
            best_prograde = float(prograde)

    for radial in range(-150, 151, 10):
        candidate_periapsis = kerbin_periapsis_after_correction(
            vessel, correction_ut, best_prograde, float(radial)
        )
        error = abs(candidate_periapsis - TARGET_KERBIN_PERIAPSIS)
        if error < best_error:
            best_error = error
            best_radial = float(radial)

    for step in (2.0, 0.5, 0.1):
        center_prograde = best_prograde
        center_radial = best_radial
        for prograde_offset in range(-3, 4):
            for radial_offset in range(-3, 4):
                prograde = center_prograde + prograde_offset * step
                radial = center_radial + radial_offset * step
                candidate_periapsis = kerbin_periapsis_after_correction(
                    vessel, correction_ut, prograde, radial
                )
                error = abs(candidate_periapsis - TARGET_KERBIN_PERIAPSIS)
                if error < best_error:
                    best_error = error
                    best_prograde = prograde
                    best_radial = radial

    node = vessel.control.add_node(
        correction_ut,
        prograde=best_prograde,
        radial=best_radial,
    )
    predicted_periapsis = node.orbit.periapsis_altitude
    if not (
        MIN_SAFE_KERBIN_PERIAPSIS - 5_000.0
        <= predicted_periapsis
        <= MAX_SAFE_KERBIN_PERIAPSIS + 5_000.0
    ):
        node.remove()
        raise RuntimeError(
            "Не удалось рассчитать безопасную коррекцию входа: "
            f"Pe={predicted_periapsis / 1000.0:.1f} км."
        )
    print(
        f"Коррекция входа: prograde={best_prograde:.1f} м/с, "
        f"radial={best_radial:.1f} м/с; "
        f"прогноз Pe={predicted_periapsis / 1000.0:.1f} км."
    )
    return node


def altitude_at_ut(orbit, ut: float) -> float:
    body = orbit.body
    position = orbit.position_at(ut, body.non_rotating_reference_frame)
    return vector_length(position) - body.equatorial_radius


def inbound_ut_for_altitude(conn, vessel, target_altitude: float) -> float:
    orbit = vessel.orbit
    current_ut = conn.space_center.ut
    if altitude_at_ut(orbit, current_ut) <= target_altitude:
        return current_ut
    high_ut = current_ut + orbit.time_to_periapsis
    if altitude_at_ut(orbit, high_ut) > target_altitude:
        raise RuntimeError(
            f"Траектория не достигает высоты {target_altitude / 1000.0:.0f} км."
        )
    low_ut = current_ut
    for _ in range(55):
        middle_ut = (low_ut + high_ut) / 2.0
        if altitude_at_ut(orbit, middle_ut) > target_altitude:
            low_ut = middle_ut
        else:
            high_ut = middle_ut
    return high_ut


def point_retrograde(vessel, autopilot, frame) -> None:
    velocity = vessel.velocity(frame)
    autopilot.target_direction = normalized(vector_scale(velocity, -1.0))


def separate_capsule(conn, vessel):
    space_center = conn.space_center
    target_ut = inbound_ut_for_altitude(
        conn, vessel, CAPSULE_SEPARATION_ALTITUDE
    )
    # Тяжёлую двигательную ступень перед отделением разворачивать не нужно.
    # Она плохо управляется и расходует почти весь оставшийся заряд. До точки
    # отделения летим полностью пассивно и ориентируем уже лёгкую капсулу.
    vessel.auto_pilot.engaged = False
    control = vessel.control
    control.throttle = 0.0
    control.sas = False
    control.rcs = False
    for engine in vessel.parts.engines:
        engine.active = False
    space_center.warp_to(max(space_center.ut, target_ut - 8.0))

    while altitude_at_ut(vessel.orbit, space_center.ut) > CAPSULE_SEPARATION_ALTITUDE:
        time.sleep(0.1)

    separator = find_capsule_separator(vessel)
    control.throttle = 0.0
    detached = separator.decouple()
    time.sleep(0.5)
    capsule = wait_for_capsule(space_center, detached)
    control = capsule.control
    control.throttle = 0.0
    control.rcs = False
    control.sas = False
    frame = capsule.orbit.body.non_rotating_reference_frame
    autopilot = capsule.auto_pilot
    charge, maximum = electric_charge(capsule)
    if charge >= MIN_ELECTRIC_CHARGE:
        autopilot.reference_frame = frame
        point_retrograde(capsule, autopilot, frame)
        autopilot.engaged = True
        wait_for_alignment(capsule, autopilot)
        print(
            "Возвращаемая капсула отделена и ориентирована "
            "теплозащитой по потоку."
        )
    else:
        autopilot.engaged = False
        arm_parachute(capsule)
        print(
            f"Капсула отделена при заряде {charge:.1f}/{maximum:.1f} EC. "
            "Включён пассивный вход: автопилот отключён, парашют взведён."
        )
    return capsule


def arm_parachute(capsule) -> None:
    """Взводит парашют заранее для пассивного спуска без электричества."""
    parachutes = capsule.parts.parachutes
    if len(parachutes) != 1:
        raise RuntimeError(
            f"После отделения найдено парашютов: {len(parachutes)}, ожидался один."
        )
    parachute = parachutes[0]
    if parachute.deployed or parachute.armed:
        return
    try:
        parachute.deploy_altitude = PARACHUTE_ARMED_DEPLOY_ALTITUDE
        parachute.arm()
    except RuntimeError as error:
        raise RuntimeError(
            "Не удалось заранее взвести парашют капсулы."
        ) from error
    print(
        f"Парашют взведён; автоматика раскроет его ниже "
        f"{PARACHUTE_ARMED_DEPLOY_ALTITUDE / 1000.0:.0f} км, когда условия станут безопасными."
    )


def deploy_parachute(capsule) -> None:
    parachutes = capsule.parts.parachutes
    if len(parachutes) != 1:
        raise RuntimeError(
            f"После отделения найдено парашютов: {len(parachutes)}, ожидался один."
        )
    parachute = parachutes[0]
    if parachute.deployed:
        return
    if "unsafe" in str(parachute.safe_state).lower():
        raise RuntimeError("Парашют пока нельзя безопасно раскрыть.")
    try:
        parachute.deploy()
    except RuntimeError as error:
        raise RuntimeError("Не удалось раскрыть парашют капсулы.") from error
    print("Парашют возвращаемой капсулы раскрыт.")


def perform_reentry(conn, capsule) -> None:
    space_center = conn.space_center
    body = capsule.orbit.body
    if body.name != "Kerbin":
        raise RuntimeError("Капсула ещё не находится у Кербина.")

    charge, maximum = electric_charge(capsule)
    powered_entry = charge >= MIN_ELECTRIC_CHARGE
    if not powered_entry:
        arm_parachute(capsule)
        print(
            f"Заряд {charge:.1f}/{maximum:.1f} EC: капсула входит "
            "в атмосферу пассивно."
        )

    approach_ut = inbound_ut_for_altitude(
        conn, capsule, ATMOSPHERE_APPROACH_ALTITUDE
    )
    capsule.auto_pilot.engaged = False
    capsule.control.sas = False
    space_center.warp_to(max(space_center.ut, approach_ut - 35.0))
    frame = body.non_rotating_reference_frame
    control = capsule.control
    autopilot = capsule.auto_pilot
    autopilot.reference_frame = frame
    control.sas = False
    autopilot.engaged = powered_entry
    print("Начинается вход в атмосферу Кербина.")

    parachute = capsule.parts.parachutes[0]
    parachute_commanded = parachute.armed or parachute.deployed
    last_report_altitude = float("inf")
    while True:
        # Сначала проверяем касание: у неподвижной капсулы уже нет вектора
        # скорости, поэтому команда ретроградного наведения ей не нужна.
        situation = str(capsule.situation).lower()
        if "landed" in situation or "splashed" in situation:
            break
        if powered_entry and not parachute_commanded:
            point_retrograde(capsule, autopilot, frame)
        flight = capsule.flight(body.reference_frame)
        altitude = max(0.0, flight.surface_altitude)
        speed = flight.speed

        # Электричество необходимо только до команды парашюту: оно питает
        # ориентацию теплозащиты и систему управления. После раскрытия капсула
        # спускается пассивно, поэтому разряд аккумулятора уже не является
        # аварией.
        if powered_entry and not parachute_commanded:
            charge, _ = electric_charge(capsule)
            if charge < MIN_ELECTRIC_CHARGE:
                powered_entry = False
                autopilot.engaged = False
                control.sas = False
                arm_parachute(capsule)
                parachute_commanded = True
                print(
                    "Заряд истощён во время входа; "
                    "перехожу на пассивный спуск."
                )

        if (
            not parachute_commanded
            and altitude <= PARACHUTE_DEPLOY_ALTITUDE
            and speed <= PARACHUTE_DEPLOY_SPEED
        ):
            deploy_parachute(capsule)
            parachute_commanded = True
            autopilot.engaged = False
            control.sas = False

        if altitude < last_report_altitude - 10_000.0:
            print(
                f"Высота {altitude / 1000.0:.1f} км; "
                f"скорость {speed:.1f} м/с."
            )
            last_report_altitude = altitude

        if altitude < 2_000.0 and not parachute_commanded:
            if speed > 350.0:
                raise RuntimeError(
                    "Капсула ниже 2 км, но скорость слишком велика для парашюта."
                )
            deploy_parachute(capsule)
            parachute_commanded = True
        time.sleep(0.1)

    autopilot.engaged = False
    control.sas = False
    final_speed = capsule.flight(body.reference_frame).speed
    print(
        f"Возвращение завершено: скорость {final_speed:.1f} м/с, "
        f"ситуация {capsule.situation}."
    )


def emergency_stop() -> None:
    try:
        conn = krpc.connect(name="Luna-16 return emergency stop")
        vessel = conn.space_center.active_vessel
        if vessel is not None:
            vessel.control.throttle = 0.0
            vessel.auto_pilot.engaged = False
            vessel.control.sas = False
        conn.close()
    except Exception:
        pass


def can_resume_kerbin_reentry(vessel) -> bool:
    """Определяет сохранение после перелёта, но до отделения капсулы."""
    if vessel.orbit.body.name != "Kerbin":
        return False
    names = {part.name for part in vessel.parts.all}
    engine_names = {engine.part.name for engine in vessel.parts.engines}
    forbidden_fragments = (
        "liquidEngineMainsail",
        "liquidEngine2-2",
        "smallRadialEngine",
    )
    return (
        "probeCoreOcto2.v2" in names
        and "parachuteSingle" in names
        and not any(
            fragment in engine_name
            for engine_name in engine_names
            for fragment in forbidden_fragments
        )
    )


def resume_kerbin_reentry(conn, vessel) -> None:
    """Продолжает миссию из сохранения Luna16_before_reentry."""
    periapsis = vessel.orbit.periapsis_altitude
    if not (
        MIN_SAFE_KERBIN_PERIAPSIS - 5_000.0
        <= periapsis
        <= MAX_SAFE_KERBIN_PERIAPSIS + 5_000.0
    ):
        raise RuntimeError(
            f"Перицентр сохранённой траектории небезопасен: "
            f"{periapsis / 1000.0:.1f} км."
        )
    charge, maximum = electric_charge(vessel)
    print(
        f"Обнаружено продолжение у Кербина: Pe={periapsis / 1000.0:.1f} км, "
        f"заряд {charge:.1f}/{maximum:.1f} EC."
    )
    if input("Для продолжения входа введите ДОМОЙ: ").strip().upper() != "ДОМОЙ":
        print("Вход в атмосферу отменён.")
        return

    if vessel.parts.engines:
        capsule = separate_capsule(conn, vessel)
    else:
        capsule = vessel
    perform_reentry(conn, capsule)


def run() -> None:
    conn = krpc.connect(name="Luna-16 return autopilot")
    space_center = conn.space_center
    vessel = space_center.active_vessel
    if vessel is None:
        conn.close()
        raise RuntimeError("Активный аппарат не найден.")

    if can_resume_kerbin_reentry(vessel):
        resume_kerbin_reentry(conn, vessel)
        conn.close()
        return

    validate_landed_vehicle(vessel)
    ascent_mass, start_twr, start_delta_v = predicted_ascent_stage_budget(vessel)
    charge, maximum_charge = electric_charge(vessel)

    print(f"Корабль: {vessel.name}")
    print(
        f"Возвращаемая ступень: 5 двигателей, стартовая масса "
        f"{ascent_mass / 1000.0:.2f} т после отделения платформы."
    )
    print(
        f"Расчётный запас возвращаемой ступени до отделения: "
        f"около {start_delta_v:.0f} м/с; TWR на Муне {start_twr:.2f}."
    )
    print(f"Аккумулятор: {charge:.1f}/{maximum_charge:.1f} EC.")
    print(
        f"План: орбита Муны {TARGET_MUN_ORBIT_ALTITUDE / 1000.0:.0f} км, "
        f"вход в атмосферу Кербина с Pe около "
        f"{TARGET_KERBIN_PERIAPSIS / 1000.0:.0f} км."
    )

    if start_delta_v < MIN_START_DELTA_V:
        conn.close()
        raise RuntimeError(
            f"Для возвращения доступно только {start_delta_v:.0f} м/с; "
            f"требуется не менее {MIN_START_DELTA_V:.0f} м/с."
        )
    if input("Для возвращения введите ДОМОЙ: ").strip().upper() != "ДОМОЙ":
        conn.close()
        print("Возвращение отменено.")
        return
    if space_center.can_save:
        space_center.quicksave()
        print("Создано быстрое сохранение перед возвращением.")

    vessel, engines = launch_from_mun(space_center, vessel)
    stage_delta_v = available_delta_v(vessel, engines)
    thrust, _ = weighted_engine_performance(engines)
    twr = thrust / max(vessel.mass * vessel.orbit.body.surface_gravity, 1.0)
    print(
        f"У апоцентра: масса {vessel.mass / 1000.0:.2f} т, "
        f"TWR {twr:.2f}, доступно около {stage_delta_v:.0f} м/с."
    )

    circularize_mun_orbit(conn, vessel, engines)
    remaining_delta_v = available_delta_v(vessel, engines)
    print(f"После выхода на орбиту осталось около {remaining_delta_v:.0f} м/с.")

    transfer_node, predicted_periapsis = plan_kerbin_transfer(
        conn, vessel
    )
    required_delta_v = transfer_node.delta_v + TRANSFER_RESERVE_DELTA_V
    if remaining_delta_v < required_delta_v:
        transfer_node.remove()
        raise RuntimeError(
            f"Для перелёта доступно {remaining_delta_v:.0f} м/с, "
            f"требуется не менее {required_delta_v:.0f} м/с с резервом."
        )
    print(
        f"Уход к Кербину через "
        f"{(transfer_node.ut - space_center.ut) / 60.0:.1f} мин; "
        f"delta-v={transfer_node.delta_v:.1f} м/с; расчётный Pe "
        f"{predicted_periapsis / 1000.0:.1f} км."
    )
    execute_node(conn, vessel, engines, transfer_node, "Уход к Кербину")
    wait_for_kerbin_soi(conn, vessel)

    print(
        f"Вход в сферу Кербина. Фактический Pe="
        f"{vessel.orbit.periapsis_altitude / 1000.0:.1f} км."
    )
    correction_node = plan_kerbin_correction(conn, vessel)
    if correction_node is not None:
        execute_node(
            conn,
            vessel,
            engines,
            correction_node,
            "Коррекция входа в атмосферу",
        )
    periapsis = vessel.orbit.periapsis_altitude
    if not (
        MIN_SAFE_KERBIN_PERIAPSIS - 5_000.0
        <= periapsis
        <= MAX_SAFE_KERBIN_PERIAPSIS + 5_000.0
    ):
        raise RuntimeError(
            f"Фактический перицентр входа небезопасен: "
            f"{periapsis / 1000.0:.1f} км."
        )
    print(f"Траектория входа подтверждена: Pe={periapsis / 1000.0:.1f} км.")

    if space_center.can_save:
        space_center.save("Luna16_before_reentry")
        print(
            "Создано отдельное сохранение Luna16_before_reentry; "
            "quicksave на поверхности Муны сохранён."
        )
    capsule = separate_capsule(conn, vessel)
    perform_reentry(conn, capsule)
    conn.close()


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        emergency_stop()
        print("\nВозвращение остановлено пользователем.")
    except Exception as error:
        emergency_stop()
        print(f"\nАВАРИЙНАЯ ОСТАНОВКА: {error}")
        raise
