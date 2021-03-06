# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: keepaway.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='keepaway.proto',
  package='keepaway',
  serialized_pb=_b('\n\x0ekeepaway.proto\x12\x08keepaway\"j\n\x06StepIn\x12\x0e\n\x06reward\x18\x01 \x02(\x01\x12\x11\n\x05state\x18\x02 \x03(\x01\x42\x02\x10\x01\x12\x13\n\x0b\x65pisode_end\x18\x03 \x02(\x08\x12\x12\n\nplayer_pid\x18\x04 \x02(\x05\x12\x14\n\x0c\x63urrent_time\x18\x05 \x02(\x01\"\x19\n\x07StepOut\x12\x0e\n\x06\x61\x63tion\x18\x01 \x02(\x05')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_STEPIN = _descriptor.Descriptor(
  name='StepIn',
  full_name='keepaway.StepIn',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='reward', full_name='keepaway.StepIn.reward', index=0,
      number=1, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='state', full_name='keepaway.StepIn.state', index=1,
      number=2, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))),
    _descriptor.FieldDescriptor(
      name='episode_end', full_name='keepaway.StepIn.episode_end', index=2,
      number=3, type=8, cpp_type=7, label=2,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='player_pid', full_name='keepaway.StepIn.player_pid', index=3,
      number=4, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='current_time', full_name='keepaway.StepIn.current_time', index=4,
      number=5, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=28,
  serialized_end=134,
)


_STEPOUT = _descriptor.Descriptor(
  name='StepOut',
  full_name='keepaway.StepOut',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='action', full_name='keepaway.StepOut.action', index=0,
      number=1, type=5, cpp_type=1, label=2,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=136,
  serialized_end=161,
)

DESCRIPTOR.message_types_by_name['StepIn'] = _STEPIN
DESCRIPTOR.message_types_by_name['StepOut'] = _STEPOUT

StepIn = _reflection.GeneratedProtocolMessageType('StepIn', (_message.Message,), dict(
  DESCRIPTOR = _STEPIN,
  __module__ = 'keepaway_pb2'
  # @@protoc_insertion_point(class_scope:keepaway.StepIn)
  ))
_sym_db.RegisterMessage(StepIn)

StepOut = _reflection.GeneratedProtocolMessageType('StepOut', (_message.Message,), dict(
  DESCRIPTOR = _STEPOUT,
  __module__ = 'keepaway_pb2'
  # @@protoc_insertion_point(class_scope:keepaway.StepOut)
  ))
_sym_db.RegisterMessage(StepOut)


_STEPIN.fields_by_name['state'].has_options = True
_STEPIN.fields_by_name['state']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
# @@protoc_insertion_point(module_scope)
