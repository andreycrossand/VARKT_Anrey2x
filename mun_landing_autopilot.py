"""Автоматическая посадка Luna-16 с низкой орбиты Муны.

Скрипт выбирает освещённый и относительно ровный участок под текущей трассой
орбиты, отделяет пустую ступень Poodle, включает только восемь посадочных
двигателей и выполняет управляемое торможение до мягкого касания.

Перед запуском корабль должен находиться на устойчивой орбите Муны. Для
фактического начала посадки требуется ввести ПОСАДКА. Скрипт делает быстрое
сохранение до отделения ступени.
"""

from __future__ import annotations

import math
import time

import krpc


MIN_ORBIT_ALTITUDE = 15_000.0
MAX_ORBIT_ALTITUDE = 120_000.0
MIN_SUN_ELEVATION = 20.0
MAX_TERRAIN_VARIATION = 90.0
SITE_SAMPLE_RADIUS_DEG = 0.15
SITE_SEARCH_STEP = 30.0
SITE_LEAD_TIME = 180.0
MIN_ELECTRIC_CHARGE = 25.0
MIN_LANDING_TWR = 1.25
HORIZONTAL_BRAKE_TARGET = 12.0
TOUCHDOWN_ALTITUDE = 4.0
TOUCHDOWN_SPEED = 1.5
ALIGNMENT_TIMEOUT = 45.0
DESCENT_BURN_MARGIN = 1.30
DESCENT_BURN_BUFFER = 250.0


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def vector_add(left, right):
    return tuple(a + b for a, b in zip(left, right))


def vector_subtract(left, right):
    return tuple(a - b for a, b in zip(left, right))


def vector_scale(vector, factor: float):
    return tuple(component * factor for component in vector)


def vector_dot(left, right) -> float:
    return sum(a * b for a, b in zip(left, right))


def vector_length(vector) -> float:
    return math.sqrt(vector_dot(vector, vector))


def normalized(vector):
    length = vector_length(vector)
    if length <= 0.0:
        raise RuntimeError("Не удалось определить направление в пространстве.")
    return tuple(component / length for component in vector)


def electric_charge(vessel) -> tuple[float, float]:
    resources = vessel.resources
    return (
        resources.amount("ElectricCharge"),
        resources.max("ElectricCharge"),
    )


def engines_with_name(vessel, name_fragment: str):
    fragment = name_fragment.lower()
    return [
        engine
        for engine in vessel.parts.engines
        if fragment in engine.part.name.lower()
    ]


def subtree_contains(part, name_fragment: str) -> bool:
    fragment = name_fragment.lower()
    pending = [part]
    while pending:
        current = pending.pop()
        if fragment in current.name.lower():
            return True
        pending.extend(current.children)
    return False


def find_poodle_separator(vessel):
    """Находит разделитель между аппаратом и пустой орбитальной ступенью."""
    candidates = []
    for part in vessel.parts.all:
        try:
            decoupler = part.decoupler
        except RuntimeError:
            continue
        if decoupler is None:
            continue
        if not subtree_contains(part, "liquidEngine2-2"):
            continue
        if subtree_contains(part, "liquidEngineMainsail"):
            continue
        candidates.append(decoupler)

    if len(candidates) != 1:
        raise RuntimeError(
            "Не удалось однозначно найти разделитель пустой ступени Poodle "
            f"(найдено: {len(candidates)})."
        )
    return candidates[0]


def sun_elevation(conn, body, latitude: float, longitude: float) -> float:
    frame = body.non_rotating_reference_frame
    site = body.surface_position(latitude, longitude, frame)
    sun = conn.space_center.bodies["Sun"].position(frame)
    normal = normalized(site)
    direction_to_sun = normalized(vector_subtract(sun, site))
    return math.degrees(math.asin(clamp(vector_dot(normal, direction_to_sun), -1.0, 1.0)))


def terrain_variation(body, latitude: float, longitude: float) -> float:
    delta = SITE_SAMPLE_RADIUS_DEG
    heights = []
    for dlat, dlon in (
        (0.0, 0.0),
        (delta, 0.0),
        (-delta, 0.0),
        (0.0, delta),
        (0.0, -delta),
        (delta, delta),
        (delta, -delta),
        (-delta, delta),
        (-delta, -delta),
    ):
        sample_latitude = clamp(latitude + dlat, -89.8, 89.8)
        sample_longitude = (longitude + dlon + 180.0) % 360.0 - 180.0
        heights.append(body.surface_height(sample_latitude, sample_longitude))
    return max(heights) - min(heights)


def find_landing_site(conn, vessel):
    """Ищет светлый ровный участок, над которым аппарат пройдёт в этот виток."""
    body = vessel.orbit.body
    frame = body.reference_frame
    now = conn.space_center.ut
    period = vessel.orbit.period
    if not math.isfinite(period) or period <= 0.0:
        raise RuntimeError("Не удалось определить период орбиты Муны.")

    best = None
    end_ut = now + min(period * 1.25, 12_000.0)
    candidate_ut = now + SITE_LEAD_TIME
    while candidate_ut <= end_ut:
        position = vessel.orbit.position_at(candidate_ut, frame)
        latitude = body.latitude_at_position(position, frame)
        longitude = body.longitude_at_position(position, frame)
        illumination = sun_elevation(conn, body, latitude, longitude)
        if illumination >= MIN_SUN_ELEVATION:
            variation = terrain_variation(body, latitude, longitude)
            if variation <= MAX_TERRAIN_VARIATION:
                score = variation - illumination * 2.0
                candidate = (score, candidate_ut, latitude, longitude, illumination, variation)
                if best is None or candidate[0] < best[0]:
                    best = candidate
        candidate_ut += SITE_SEARCH_STEP

    if best is None:
        raise RuntimeError(
            "На текущем витке не найден достаточно светлый и ровный участок. "
            "Попробуйте запустить скрипт через один виток."
        )
    _, site_ut, latitude, longitude, illumination, variation = best
    return site_ut, latitude, longitude, illumination, variation


def point_opposite_velocity(vessel, autopilot, frame) -> tuple[float, float]:
    position = vessel.position(frame)
    velocity = vessel.velocity(frame)
    up = normalized(position)
    vertical_speed = vector_dot(velocity, up)
    horizontal = vector_subtract(velocity, vector_scale(up, vertical_speed))
    horizontal_speed = vector_length(horizontal)
    if vector_length(velocity) <= 0.1:
        direction = up
    else:
        direction = normalized(vector_scale(velocity, -1.0))
    autopilot.target_direction = direction
    return vertical_speed, horizontal_speed


def wait_for_alignment(vessel, autopilot) -> None:
    deadline = time.monotonic() + ALIGNMENT_TIMEOUT
    while time.monotonic() < deadline:
        charge, _ = electric_charge(vessel)
        if charge < MIN_ELECTRIC_CHARGE:
            raise RuntimeError("Недостаточно электричества для наведения.")
        if autopilot.error <= 3.0:
            return
        time.sleep(0.1)
    raise RuntimeError("Аппарат не успел развернуться к направлению торможения.")


def brake_with_poodle(vessel, autopilot, frame) -> None:
    """Использует весь доступный остаток Poodle до отделения его ступени."""
    engines = engines_with_name(vessel, "liquidEngine2-2")
    if len(engines) != 1:
        raise RuntimeError(
            f"Перед посадкой ожидался один Poodle, найдено: {len(engines)}."
        )
    engine = engines[0]
    if not engine.has_fuel:
        print("Ступень Poodle пуста; отделяю её без дополнительного импульса.")
        return
    if not engine.active:
        engine.active = True

    control = vessel.control
    control.throttle = 0.0
    autopilot.reference_frame = frame
    point_opposite_velocity(vessel, autopilot, frame)
    autopilot.engaged = True
    wait_for_alignment(vessel, autopilot)
    print("Использую остаток топлива Poodle для начального торможения.")
    control.throttle = 1.0
    thrust_deadline = time.monotonic() + 3.0

    while engine.has_fuel:
        vertical_speed, horizontal_speed = point_opposite_velocity(
            vessel, autopilot, frame
        )
        if horizontal_speed <= HORIZONTAL_BRAKE_TARGET:
            break
        if vessel.thrust < 1.0 and time.monotonic() >= thrust_deadline:
            control.throttle = 0.0
            raise RuntimeError("Poodle не создал тягу перед посадкой.")
        if vessel.flight(vessel.orbit.body.reference_frame).surface_altitude < 8_000.0:
            print(
                "Высота ниже 8 км; прекращаю работу Poodle и передаю "
                "торможение посадочной ступени."
            )
            break
        time.sleep(0.05)

    control.throttle = 0.0
    _, horizontal_speed = point_opposite_velocity(vessel, autopilot, frame)
    print(
        f"Работа Poodle завершена; горизонтальная скорость "
        f"{horizontal_speed:.1f} м/с."
    )


def landing_direction(vessel, frame, horizontal_speed: float):
    position = vessel.position(frame)
    velocity = vessel.velocity(frame)
    up = normalized(position)
    vertical_speed = vector_dot(velocity, up)
    horizontal = vector_subtract(velocity, vector_scale(up, vertical_speed))
    if horizontal_speed <= 0.2:
        return up
    horizontal_opposite = normalized(vector_scale(horizontal, -1.0))
    tilt = clamp(horizontal_speed / 45.0, 0.0, 0.45)
    return normalized(
        vector_add(
            vector_scale(up, 1.0 - tilt),
            vector_scale(horizontal_opposite, tilt),
        )
    )


def target_descent_speed(radar_altitude: float) -> float:
    if radar_altitude > 15_000.0:
        return -85.0
    if radar_altitude > 8_000.0:
        return -65.0
    if radar_altitude > 4_000.0:
        return -45.0
    if radar_altitude > 2_000.0:
        return -30.0
    if radar_altitude > 800.0:
        return -18.0
    if radar_altitude > 300.0:
        return -10.0
    if radar_altitude > 100.0:
        return -6.0
    if radar_altitude > 30.0:
        return -3.0
    return -0.8


def activate_landing_engines(vessel):
    engines = engines_with_name(vessel, "smallRadialEngine")
    if len(engines) != 8:
        raise RuntimeError(
            f"Ожидалось восемь посадочных двигателей, найдено: {len(engines)}."
        )
    for engine in engines:
        if not engine.has_fuel:
            raise RuntimeError("Посадочный двигатель не имеет доступного топлива.")
        if not engine.active:
            engine.active = True
    return engines


def deploy_landing_legs(vessel) -> None:
    legs = vessel.parts.legs
    if not legs:
        raise RuntimeError("Посадочные опоры не найдены.")
    for leg in legs:
        if leg.deployable and not leg.deployed:
            leg.deployed = True
    print(f"Посадочные опоры раскрыты: {len(legs)} шт.")


def emergency_stop() -> None:
    try:
        conn = krpc.connect(name="Luna-16 landing emergency stop")
        vessel = conn.space_center.active_vessel
        if vessel is not None:
            vessel.control.throttle = 0.0
            vessel.auto_pilot.engaged = False
            vessel.control.sas = True
        conn.close()
    except Exception:
        pass


def run() -> None:
    conn = krpc.connect(name="Luna-16 Mun landing autopilot")
    space_center = conn.space_center
    vessel = space_center.active_vessel
    if vessel is None:
        conn.close()
        raise RuntimeError("Активный корабль не найден.")

    body = vessel.orbit.body
    if body.name != "Mun":
        conn.close()
        raise RuntimeError("Посадочный автопилот запускается только у Муны.")
    if vessel.orbit.eccentricity >= 1.0:
        conn.close()
        raise RuntimeError("Сначала сформируйте связанную орбиту Муны.")
    if not (
        MIN_ORBIT_ALTITUDE <= vessel.orbit.periapsis_altitude
        and vessel.orbit.apoapsis_altitude <= MAX_ORBIT_ALTITUDE
    ):
        conn.close()
        raise RuntimeError(
            "Для посадки нужна устойчивая низкая орбита примерно 15–120 км."
        )

    charge, maximum = electric_charge(vessel)
    if charge < MIN_ELECTRIC_CHARGE:
        conn.close()
        raise RuntimeError("Недостаточно электричества для посадки.")

    landing_engines = engines_with_name(vessel, "smallRadialEngine")
    if len(landing_engines) != 8:
        conn.close()
        raise RuntimeError(
            f"Конструкция не прошла проверку: посадочных двигателей "
            f"найдено {len(landing_engines)}, ожидалось 8."
        )

    site_ut, latitude, longitude, illumination, variation = find_landing_site(
        conn, vessel
    )
    print(f"Корабль: {vessel.name}")
    print(
        f"Орбита Муны: Pe={vessel.orbit.periapsis_altitude / 1000.0:.1f} км, "
        f"Ap={vessel.orbit.apoapsis_altitude / 1000.0:.1f} км."
    )
    print(f"Аккумулятор: {charge:.1f}/{maximum:.1f} EC.")
    print(
        f"Выбран светлый участок: широта {latitude:.2f}°, "
        f"долгота {longitude:.2f}°, Солнце {illumination:.1f}° над горизонтом, "
        f"перепад рельефа около {variation:.0f} м."
    )
    print(f"До начала посадки: {(site_ut - space_center.ut) / 60.0:.1f} мин.")

    if input("Для начала посадки введите ПОСАДКА: ").strip().upper() != "ПОСАДКА":
        conn.close()
        print("Посадка отменена.")
        return

    if space_center.can_save:
        space_center.quicksave()
        print("Создано быстрое сохранение перед посадкой.")

    space_center.warp_to(max(space_center.ut, site_ut - 90.0))
    while space_center.ut < site_ut:
        time.sleep(0.05)

    control = vessel.control
    control.throttle = 0.0
    frame = body.reference_frame
    autopilot = vessel.auto_pilot
    brake_with_poodle(vessel, autopilot, frame)

    separator = find_poodle_separator(vessel)
    separator.decouple()
    time.sleep(1.5)
    vessel = space_center.active_vessel
    control = vessel.control
    autopilot = vessel.auto_pilot

    engines = activate_landing_engines(vessel)
    deploy_landing_legs(vessel)
    max_thrust = sum(engine.max_vacuum_thrust for engine in engines)
    local_gravity = body.surface_gravity
    twr = max_thrust / max(vessel.mass * local_gravity, 1.0)
    print(
        f"Посадочная ступень отделена. Масса {vessel.mass / 1000.0:.2f} т, "
        f"тяговооружённость на Муне {twr:.2f}."
    )
    if twr < MIN_LANDING_TWR:
        raise RuntimeError(
            f"Недостаточная тяговооружённость для посадки: {twr:.2f}."
        )

    autopilot.reference_frame = frame
    point_opposite_velocity(vessel, autopilot, frame)
    autopilot.engaged = True
    wait_for_alignment(vessel, autopilot)
    print("Начинаю гашение орбитальной скорости.")

    zero_thrust_since = None
    while True:
        vertical_speed, horizontal_speed = point_opposite_velocity(
            vessel, autopilot, frame
        )
        flight = vessel.flight(body.reference_frame)
        radar_altitude = flight.surface_altitude
        control.throttle = 1.0
        if horizontal_speed <= HORIZONTAL_BRAKE_TARGET:
            break
        if vessel.thrust < 1.0:
            if zero_thrust_since is None:
                zero_thrust_since = time.monotonic()
            elif time.monotonic() - zero_thrust_since >= 1.0:
                control.throttle = 0.0
                if not any(engine.has_fuel for engine in engines):
                    raise RuntimeError(
                        "Посадочные двигатели выработали топливо во время "
                        "гашения орбитальной скорости."
                    )
                raise RuntimeError(
                    "Посадочные двигатели не создают тягу во время торможения."
                )
        else:
            zero_thrust_since = None
        if radar_altitude < 6_000.0 and vertical_speed < -90.0:
            print(
                "Высота ниже 6 км: перехожу к управляемому снижению с "
                f"остаточной горизонтальной скоростью {horizontal_speed:.1f} м/с."
            )
            break
        time.sleep(0.05)

    control.throttle = 0.0
    print(
        "Перехожу к экономичному снижению: свободное падение до расчётной "
        "точки торможения."
    )
    previous_report_altitude = float("inf")
    zero_thrust_since = None
    while True:
        flight = vessel.flight(body.reference_frame)
        radar_altitude = max(0.0, flight.surface_altitude)
        vertical_speed = flight.vertical_speed
        horizontal_speed = flight.horizontal_speed
        autopilot.target_direction = landing_direction(
            vessel, frame, horizontal_speed
        )

        target_speed = target_descent_speed(radar_altitude)
        downward_speed = max(0.0, -vertical_speed)
        target_downward_speed = max(0.0, -target_speed)
        maximum_acceleration = vessel.available_thrust / max(vessel.mass, 1.0)
        braking_acceleration = max(
            0.1,
            maximum_acceleration - local_gravity,
        )
        speed_to_remove_squared = max(
            0.0,
            downward_speed**2 - target_downward_speed**2,
        )
        stopping_distance = (
            speed_to_remove_squared / (2.0 * braking_acceleration)
        )
        ignition_altitude = (
            stopping_distance * DESCENT_BURN_MARGIN
            + DESCENT_BURN_BUFFER
        )

        if radar_altitude > ignition_altitude and radar_altitude > 300.0:
            # Частичная тяга на большой высоте только растягивает торможение и
            # тратит топливо на компенсацию гравитации. Здесь выгоднее падать.
            control.throttle = 0.0
        elif vertical_speed < target_speed - 2.0:
            # Короткий мощный импульс заметно экономичнее долгого зависания.
            control.throttle = 1.0
        else:
            speed_error = target_speed - vertical_speed
            desired_acceleration = local_gravity + speed_error
            hover_throttle = (
                vessel.mass * max(0.0, desired_acceleration)
                / max(vessel.available_thrust, 1.0)
            )
            control.throttle = clamp(hover_throttle, 0.0, 1.0)

        if control.throttle > 0.05 and vessel.thrust < 1.0:
            if zero_thrust_since is None:
                zero_thrust_since = time.monotonic()
            elif time.monotonic() - zero_thrust_since >= 1.0:
                control.throttle = 0.0
                if not any(engine.has_fuel for engine in engines):
                    raise RuntimeError(
                        "Посадочные двигатели выработали топливо во время снижения."
                    )
                raise RuntimeError(
                    "Посадочные двигатели потеряли тягу во время снижения."
                )
        else:
            zero_thrust_since = None

        charge, _ = electric_charge(vessel)
        if charge < MIN_ELECTRIC_CHARGE:
            control.throttle = 0.0
            raise RuntimeError("Заряд стал критическим во время посадки.")
        if radar_altitude < previous_report_altitude - 1_000.0:
            print(
                f"Высота {radar_altitude / 1000.0:.1f} км; "
                f"Vверт={vertical_speed:.1f} м/с, "
                f"Vгор={horizontal_speed:.1f} м/с; "
                f"расчётный тормозной путь={stopping_distance / 1000.0:.1f} км."
            )
            previous_report_altitude = radar_altitude

        landed = vessel.situation == space_center.VesselSituation.landed
        if landed or (
            radar_altitude <= TOUCHDOWN_ALTITUDE
            and abs(vertical_speed) <= TOUCHDOWN_SPEED
        ):
            break
        if radar_altitude < 25.0 and vertical_speed < -5.0:
            control.throttle = 1.0
        time.sleep(0.05)

    control.throttle = 0.0
    autopilot.engaged = False
    control.sas = True
    time.sleep(2.0)
    flight = vessel.flight(body.reference_frame)
    print(
        f"Посадка завершена: широта {flight.latitude:.3f}°, "
        f"долгота {flight.longitude:.3f}°."
    )
    liquid_fuel = vessel.resources.amount("LiquidFuel")
    oxidizer = vessel.resources.amount("Oxidizer")
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
        print("\nПосадка остановлена пользователем.")
    except Exception as error:
        emergency_stop()
        print(f"\nАВАРИЙНАЯ ОСТАНОВКА: {error}")
        raise
