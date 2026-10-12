# Offline check only. No board files, project, or bitstream required.
set target_part [lindex $argv 0]
set found [get_parts -quiet $target_part]
if {[llength $found] != 1} {
    error "Target part unavailable: $target_part"
}
puts "PART_CHECK_PASSED: $found"
exit
