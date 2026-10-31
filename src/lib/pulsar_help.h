/** @file
 *  The command-line help printers of every pulsar binary. */
#ifndef PULSAR_HELP_H
#define PULSAR_HELP_H

#include <stdio.h>


/// The binary whose help is printed.
typedef enum {
    /// `pulsar`, the CLI
    PULSAR_HELP_DS4,
    /// `pulsar-server`
    PULSAR_HELP_SERVER,
    /// `pulsar-agent`
    PULSAR_HELP_AGENT,
    /// `pulsar-bench`
    PULSAR_HELP_BENCH,
    /// `pulsar-eval`
    PULSAR_HELP_EVAL,
} pulsar_help_tool;

/// Print `tool`'s help to `fp`: name, summary and usage, then the default sections (or one topic's), then examples.
/// An unknown topic is reported on `fp` and the default help follows.
/// @param fp    the sink (colour only when it is a TTY)
/// @param tool  the binary
/// @param topic the `--help TOPIC` requested, or NULL for the default help
void pulsar_help_print(FILE *fp, pulsar_help_tool tool, const char *topic);


#endif
