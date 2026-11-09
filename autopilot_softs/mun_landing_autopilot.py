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
MIN_SUN_ELEVATION = 35.0
PREFERRED_TERRAIN_VARIATION = 35.0
SITE_SAMPLE_RADIUS_DEG = 0.35
SITE_SEARCH_STEP = 30.0
SITE_LEAD_TIME = 180.0
MIN_ELECTRIC_CHARGE = 25.0
MIN_LANDING_TWR = 1.25
POODLE_BRAKE_TARGET_SPEED = 5.0
TOUCHDOWN_HORIZONTAL_SPEED = 2.0
TOUCHDOWN_ALTITUDE = 4.0
TOUCHDOWN_SPEED = 1.5
ALIGNMENT_TIMEOUT = 75.0
DESCENT_BURN_MARGIN = 1.12
DESCENT_BURN_BUFFER = 100.0
POODLE_SEPARATION_FORCE_PERCENT = 20.0
POODLE_CLEARANCE_DISTANCE = 5.0
POODLE_CLEARANCE_TIMEOUT = 8.0


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


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
    sine = clamp(vector_dot(normal, direction_to_sun), -1.0, 1.0)
    return math.degrees(math.asin(sine))


def terrain_variation(body, latitude: float, longitude: float) -> float:
    """Измеряет перепад высот на широкой сетке вокруг точки посадки."""
    delta = SITE_SAMPLE_RADIUS_DEG
    heights = []
    offsets = (-delta, -delta / 2.0, 0.0, delta / 2.0, delta)
    for dlat in offsets:
        for dlon in offsets:
            sample_latitude = clamp(latitude + dlat, -89.8, 89.8)
            sample_longitude = (longitude + dlon + 180.0) % 360.0 - 180.0
            heights.append(
                body.surface_height(sample_latitude, sample_longitude)
            )
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
            # Ровность — главный критерий. Жёстко отбрасывать площадки нельзя:
            # на некоторых трассах Муны идеального участка просто нет.
            candidate = (
                variation,
                -illumination,
                candidate_ut,
                latitude,
                longitude,
                illumination,
            )
            if best is None or candidate < best:
                best = candidate
        candidate_ut += SITE_SEARCH_STEP

    if best is None:
        raise RuntimeError(
            "На доступной трассе не найден ни один освещённый участок. "
            "Попробуйте запустить скрипт через один виток."
        )
    variation, _, site_ut, latitude, longitude, illumination = best
    if variation > PREFERRED_TERRAIN_VARIATION:
        print(
            f"Предупреждение: участка с перепадом до "
            f"{PREFERRED_TERRAIN_VARIATION:.0f} м на трассе нет; "
            f"выбран самый ровный доступный ({variation:.0f} м)."
        )
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
        if autopilot.error <= 5.0:
            return
        time.sleep(0.1)
    raise RuntimeError(
        "Аппарат не успел развернуться к направлению торможения "
        f"за {ALIGNMENT_TIMEOUT:.0f} с."
    )


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
        total_speed = math.hypot(vertical_speed, horizontal_speed)
        if total_speed <= POODLE_BRAKE_TARGET_SPEED:
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
    # Перед разделением гарантированно глушим двигатель. Иначе остаточная тяга
    # и работа автопилота могут впечатать Poodle в посадочный аппарат.
    engine.active = False
    vertical_speed, horizontal_speed = point_opposite_velocity(
        vessel, autopilot, frame
    )
    print(
        f"Работа Poodle завершена; скорость: горизонтальная "
        f"{horizontal_speed:.1f} м/с, вертикальная {vertical_speed:.1f} м/с."
    )


def wait_for_lander(space_center, timeout: float = 10.0):
    """Ждёт именно отделившийся модуль, а не старую составную связку."""
    deadline = time.monotonic() + timeout
    stable_reads = 0
    candidate = None
    while time.monotonic() < deadline:
        try:
            candidate = space_center.active_vessel
            if candidate is None:
                stable_reads = 0
                time.sleep(0.1)
                continue
            landing_engines = engines_with_name(candidate, "smallRadialEngine")
            poodle_engines = engines_with_name(candidate, "liquidEngine2-2")
            _ = candidate.mass
            # До завершения физического кадра KSP active_vessel уже доступен,
            # но всё ещё может содержать Poodle. Такой объект принимать нельзя.
            if len(landing_engines) == 8 and not poodle_engines:
                stable_reads += 1
                if stable_reads >= 5:
                    return candidate
            else:
                stable_reads = 0
        except Exception as error:
            if error.__class__.__name__ != "ObjectNotLoadedException":
                raise
            stable_reads = 0
        time.sleep(0.1)
    raise RuntimeError(
        "После отделения Poodle не удалось устойчиво выбрать посадочный модуль."
    )


def reduce_separator_force(separator) -> None:
    """Уменьшает резкий толчок разделителя перед отделением тяжёлого модуля."""
    try:
        module = next(
            item
            for item in separator.part.modules
            if item.name == "ModuleDecouple"
        )
        field = next(
            item
            for item in module.field_list
            if item.name == "ejectionForcePercent"
        )
        field.float_value = POODLE_SEPARATION_FORCE_PERCENT
        print(
            f"Сила разделителя Poodle снижена до "
            f"{POODLE_SEPARATION_FORCE_PERCENT:.0f}%."
        )
    except (RuntimeError, StopIteration):
        # В старых версиях kRPC это внутреннее поле может быть недоступно.
        # Fixed 2 содержит такое же ограничение непосредственно в craft-файле.
        print(
            "Предупреждение: kRPC не позволил изменить силу разделителя; "
            "использую штатное значение конструкции."
        )


def wait_for_poodle_clearance(lander, poodle_stage, frame) -> None:
    """Не разрешает полное торможение, пока ступень Poodle находится рядом."""
    if poodle_stage is None:
        time.sleep(1.0)
        return

    deadline = time.monotonic() + POODLE_CLEARANCE_TIMEOUT
    last_distance = 0.0
    while time.monotonic() < deadline:
        try:
            separation = vector_subtract(
                lander.position(frame),
                poodle_stage.position(frame),
            )
            last_distance = vector_length(separation)
            if last_distance >= POODLE_CLEARANCE_DISTANCE:
                print(f"Расстояние до ступени Poodle: {last_distance:.1f} м.")
                return
        except RuntimeError:
            # Ступень уже вышла из физического диапазона или была уничтожена.
            return
        time.sleep(0.1)
    raise RuntimeError(
        f"Ступень Poodle не отошла на безопасные "
        f"{POODLE_CLEARANCE_DISTANCE:.0f} м; расстояние {last_distance:.1f} м."
    )


def safely_separate_poodle(space_center, vessel, separator):
    """Гасит тягу и мягко отводит посадочный аппарат от ступени Poodle."""
    control = vessel.control
    autopilot = vessel.auto_pilot
    control.throttle = 0.0
    control.sas = False
    control.rcs = False

    # Сохраняем точное ретроградное положение всей связки до отделения. Смена
    # kRPC-autopilot на SAS в этот момент раньше давала короткий рывок по крену.
    frame = vessel.orbit.body.reference_frame
    autopilot.reference_frame = frame
    point_opposite_velocity(vessel, autopilot, frame)
    autopilot.engaged = True
    wait_for_alignment(vessel, autopilot)

    # Дожидаемся двух спокойных физических кадров с нулевой тягой.
    quiet_samples = 0
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and quiet_samples < 2:
        if vessel.thrust < 1.0:
            quiet_samples += 1
        else:
            quiet_samples = 0
        time.sleep(0.1)
    if quiet_samples < 2:
        raise RuntimeError("Poodle не успел полностью погасить тягу перед отделением.")

    reduce_separator_force(separator)
    detached_vessel = separator.decouple()
    vessel = wait_for_lander(space_center)
    control = vessel.control
    control.throttle = 0.0
    control.sas = False

    if engines_with_name(vessel, "liquidEngine2-2"):
        raise RuntimeError(
            "После команды отделения Poodle всё ещё принадлежит посадочному "
            "модулю. Включение посадочных двигателей запрещено."
        )

    # Сразу возвращаем ретроградное наведение новому объекту, но держим тягу
    # закрытой до окончания разворота. Так двигатели не расходуют delta-v боком.
    autopilot = vessel.auto_pilot
    autopilot.reference_frame = frame
    point_opposite_velocity(vessel, autopilot, frame)
    control.rcs = bool(vessel.parts.rcs)
    autopilot.engaged = True
    wait_for_alignment(vessel, autopilot)

    poodle_stage = None
    try:
        if engines_with_name(detached_vessel, "liquidEngine2-2"):
            poodle_stage = detached_vessel
    except RuntimeError:
        pass
    if poodle_stage is None:
        for candidate in space_center.vessels:
            try:
                if engines_with_name(candidate, "liquidEngine2-2"):
                    poodle_stage = candidate
                    break
            except RuntimeError:
                continue

    # После выравнивания даём около 1 м/с осевого отхода. Это несравнимо
    # дешевле, чем терять десятки м/с на торможении раскрученного аппарата.
    landing_engines = activate_landing_engines(vessel)
    control.throttle = 0.15
    time.sleep(0.60)
    control.throttle = 0.0
    wait_for_poodle_clearance(vessel, poodle_stage, frame)
    control.rcs = False
    print("Отделение Poodle подтверждено; ступени безопасно разошлись.")
    return vessel


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
    vessel = safely_separate_poodle(space_center, vessel, separator)
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
    # Stability Assist, оставшийся после разделения, может бороться с kRPC и
    # удерживать прежнее положение. Наведение выполняет только автопилот.
    control.sas = False
    control.rcs = bool(vessel.parts.rcs)
    autopilot.engaged = True
    wait_for_alignment(vessel, autopilot)
    control.rcs = False
    print(
        "Начинаю совмещённое торможение: горизонтальная и вертикальная "
        "скорости гасятся одним ретроградным импульсом."
    )
    previous_report_altitude = float("inf")
    zero_thrust_since = None
    while True:
        vertical_speed, horizontal_speed = point_opposite_velocity(
            vessel, autopilot, frame
        )
        flight = vessel.flight(body.reference_frame)
        radar_altitude = max(0.0, flight.surface_altitude)

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
            downward_speed**2
            + horizontal_speed**2
            - target_downward_speed**2,
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
        elif (
            vertical_speed < target_speed - 2.0
            or horizontal_speed > TOUCHDOWN_HORIZONTAL_SPEED
        ):
            # Один короткий ретроградный импульс одновременно гасит обе
            # составляющие скорости. Это дешевле двух отдельных торможений.
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
            and horizontal_speed <= TOUCHDOWN_HORIZONTAL_SPEED
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
